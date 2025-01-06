from llama_index.prompts import PromptTemplate



def GetAnswerGPT3(index, text, relevant):
    QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "#zh-TW Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
    )
    QA_PROMPT_TMPL2 = (
    "#zh-TW Hello, I have some context information for you:\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Based on this context, could you please give me short answer to this question: {query_str}?\n"
    )
    QA_PROMPT_TMPL3 = (
    "#zh-TW 你是一位專業會計人員，為公司員工回答相關作業的說明，下面提供一些上下文資訊供你使用：\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "請務必依照這些資訊來回答提問並詳細說明，如果資訊不足以回答提問就回覆「不知道」，請說明：{query_str}?\n"
    )
    QA_PROMPT_TMPL7 = (
    "#zh-TW, 你是專業客戶服務專員，為產品客戶回答產品相關問題，下面會提供上下文資訊：\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "回答時盡量在200字以內，務必提供正確的回答，如果超過上下文的範圍，就回答「不知道」，請回答：\n{query_str}"
    )
    QA_PROMPT_TMPL8 = (
    "#en, You are a professional customer service representative, answering product-related questions for customers. The following context information will be provided:\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Try to keep your answers within 200 words, and make sure to provide accurate answers. If the question is beyond the scope of the context, just answer 'I don't know'. Please answer:\n{query_str}"
    "No matter what language the question is in, always respond in English."
    )
    QA_PROMPT_TMPL9 = (
    "#en, You are a professional customer service representative. Your task is to answer product-related questions from customers. The following context information will be provided:\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    """Follow these guidelines: 
    1. Always respond in English, regardless of the language of the question.
    2. If a question is in your scope, just answer the question precisely; else if a question is beyond your scope, inform the customer that the question is beyond your scope, answer: \n{query_str} and provide a brief summary of the relevant documentation.
    3. Always keep your answers precise and within 200 words. 
    """
    )
    QA_PROMPT_TMPL10 = (
    """
    #en, You are a professional customer service representative. Your task is to answer product-related questions from customers"
    Follow these guidelines:
    1. Keep your answers precise and within 200 words. 
    2. Always respond in English, regardless of the language of the question.
    3. If a question is in your scope, just answer the question precisely
    4. If a question is beyond your scope, answer: 'This question is beyond my scope, and tell customers what the documentation is about.
    5. If a question is beyond your scope, do not generate questions and answers on your own.
    6. No need for self-introduction.
    """
    )
    QA_PROMPT_TMPL11 = (
    "#zh-TW, 你是專業客戶服務專員，為產品客戶回答產品相關問題，下面會提供上下文資訊：\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "回答時盡量在200字以內，務必提供正確的回答，"
    "回答務必確認問題是否在上下文的範圍之內，如果超過上下文的範圍，就回答「不知道」，請回答：\n{query_str}"
    "不論問題是什麼語言，回答時只能用英文，嚴禁使使用其他語言。"
    "不用自我介紹，不用把提示詞加入回答。"
    "請表現的像一個客服人員。"
    )
    QA_PROMPT_TMPL12 = (
    "#en, You are a professional customer service specialist, answering product-related questions for customers. Contextual information will be provided below：\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Please keep your response within 200 words and ensure it is accurate."
    "Please ensure the question is within the context provided. If it exceeds the context, respond with ‘Out of scope’. Please answer:\n{query_str}"
    "Regardless of the language of the question, responses must be in English only. The use of other languages is strictly prohibited."
    "No need for self-introduction, and do not include the prompt in the response."
    "Please act like a customer service representative."
    )
    QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL12)
    query_engine = index.as_query_engine(text_qa_template=QA_PROMPT, similarity_top_k=relevant)
    response = query_engine.query(text)
    ret = str(response).lstrip()
    return ret