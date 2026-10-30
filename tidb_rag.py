import os
import pandas as pd

from litellm import completion
import streamlit as st

from typing import Optional, Any
from pytidb import TiDBClient
from pytidb.schema import TableModel, Field
from pytidb.embeddings import EmbeddingFunction

from common import LLM_RESPONSE_STYLE, RAG_PROMPT_TEMPLATE


db = TiDBClient.connect(
    host=os.getenv("SERVERLESS_CLUSTER_HOST"),
    port=int(os.getenv("SERVERLESS_CLUSTER_PORT")),
    username=os.getenv("SERVERLESS_CLUSTER_USERNAME"),
    password=os.getenv("SERVERLESS_CLUSTER_PASSWORD"),
    database=os.getenv("SERVERLESS_CLUSTER_DATABASE_NAME"),
    enable_ssl=True,
)

embedding_model = "bedrock/amazon.titan-embed-text-v2:0"
llm_model = "bedrock/us.amazon.nova-pro-v1:0"

text_embedding_function = EmbeddingFunction(
    embedding_model,
    timeout=60
)

table_name = "employees"
class Employee(TableModel, table=True):
    # adaptor for streamlit
    __tablename__ = table_name
    __table_args__ = {"extend_existing": True}
    id: int | None = Field(default=None, primary_key=True)
    name: str
    intro: str
    intro_vec: Optional[Any] = text_embedding_function.VectorField(
        source_field="intro",
    )

# Test data
employees = [
    Employee(id=1, name="Cheese Wang", intro="Cheese is a Senior Software Engineer of AI Application at PingCAP, the company behind TiDB"),
    Employee(id=2, name="Hao Huo", intro="Hao Huo is an engineer of PingCAP, the company behind TiDB | Build pingcap/autoflow / OSSInsight.io"),
    Employee(id=3, name="Humble BGL", intro="Humble BGL is the Head of Engagement Innovation at PingCAP, the company behind TiDB")
]

table = db.open_table(table_name)
if table is None:
    table = db.create_table(schema=Employee)

# has_fts_index() always returns False on TiDB Cloud Starter (https://github.com/pingcap/pytidb/issues/264),
# so the second run would fail with "Duplicate key name". Check the schema instead.
if not db.query(f"SHOW INDEXES FROM {table_name} WHERE Column_name = 'intro' AND Index_type = 'FULLTEXT'").to_list():
    table.create_fts_index("intro")


st.title("🔍 TiDB RAG Demo")

st.subheader("Database Operations")

left, right = st.columns(2)

if left.button("Reset", type="primary"):
    left.write("Resetting...")
    table.truncate()
    st.rerun()

left.markdown("This option <span style='color: red;'>will delete all data</span></h3>", unsafe_allow_html=True)

if right.button("Save Data"):
    right.write("Saving...")
    # insert sample employees
    if table.rows() == 0:
        table.bulk_insert(employees)
        db.execute("ALTER TABLE employees COMPACT")
        st.rerun()

with st.expander("📁 All Employees in the Database", expanded=False):
    # The first 100 rows only: table.query() fetches every row with its vector, on every rerun.
    employees = db.query(f"SELECT id, name, intro, intro_vec FROM {table_name} ORDER BY id LIMIT 100").to_list()
    if employees:
        st.caption(f"First {len(employees)} of {table.rows()} rows")
        data = [{'id': e['id'], 'name': e['name'], 'intro': e['intro'], 'intro_vec': e['intro_vec']} for e in employees]
        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        st.info("No data found in database.")

st.write(
    "Enter your question, and the system will retrieve relevant knowledge and generate an answer"
)
mode = st.radio("Select Mode:", ["Retrieval Only", "RAG Q&A"])

query = st.text_input("Enter your question:", "")

if st.button("Send") and query:
    with st.spinner("Processing..."):
        # Retrieve relevant employees
        results = table.search(query, search_type="hybrid").text_column("intro").limit(int(os.getenv("RAG_LIMIT", "3")))

        if results:
            if mode == "Retrieval Only":
                st.write("### Hybrid Search Results:")
                st.dataframe(results.to_pandas())
            else:
                intro = [employee.intro for employee in results.to_rows()]

                # Build RAG prompt
                context = "\n".join(intro)
                prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

                # Call LLM to generate answer
                response = completion(
                    model=llm_model,
                    messages=[{"content": prompt, "role": "user"}],
                    # We use AWS Inference Profile here to track costs.
                    # If you are using this code outside of TiDB Labs, you need to remove the line below.
                    model_id="arn:aws:bedrock:us-west-2:841162690310:application-inference-profile/4i83xkynoouo",
                )

                st.markdown(f"### 🤖 {llm_model}")
                st.markdown(LLM_RESPONSE_STYLE, unsafe_allow_html=True)

                # show the response
                st.markdown(
                    f'<div class="llm-response">{response.choices[0].message.content}</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("📚 Hybrid Search Retrieved Knowledge", expanded=False):
                    st.dataframe(results.to_pandas())
        else:
            st.info("No relevant information found")