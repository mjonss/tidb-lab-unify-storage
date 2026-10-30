import os
from litellm import completion
import streamlit as st

from common import LLM_RESPONSE_STYLE

llm_model = "bedrock/us.amazon.nova-pro-v1:0"

st.title("🔍 Vanilla AI Q&A Demo")


st.write(
    "Enter your question, and the system will answer it based on the built-in knowledge"
)

query = st.text_input("Enter your question:", "")

if st.button("Send") and query:
    with st.spinner("Processing..."):
        prompt = f"Answer the question: {query}"

        # Call LLM to generate answer
        response = completion(
            model=llm_model,
            messages=[{"content": prompt, "role": "user"}],
            # To track costs with an AWS application inference profile, as TiDB Labs do, set BEDROCK_MODEL_ID to its ARN.
            # Without it, the model is called directly with your own AWS credentials.
            **({"model_id": os.environ["BEDROCK_MODEL_ID"]} if os.getenv("BEDROCK_MODEL_ID") else {}),
        )

        st.markdown(f"### 🤖 {llm_model}")
        st.markdown(LLM_RESPONSE_STYLE, unsafe_allow_html=True)

        # show the response
        st.markdown(
            f'<div class="llm-response">{response.choices[0].message.content}</div>',
            unsafe_allow_html=True,
        )