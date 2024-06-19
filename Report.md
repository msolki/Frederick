# Fredrick

**Virtual AI Assistant for University of Naples Federico II**

<div style="text-align: center;"><img src="https://upload.wikimedia.org/wikipedia/commons/5/5c/University_Federico_II_Logo.svg" alt="UNINA Logo" height=200px width=200px></div>

*Text Mining Course Project* <br> by **Mohammad Solki**

*A.Y. 2023/24*

---


**Table of Contents**

1. [Introduction](#introduction)
2. [Background: The Rise of AI Chatbots](#background-the-rise-of-ai-chatbots)
3. [Frederick: The Virtual Assistant for UNINA Students](#frederick-the-virtual-assistant-for-unina-students)
4. [Methodology](#methodology)
   1. [Data Scraping](#data-scraping)
   2. [RAG Pipeline](#rag-pipeline)
   3. [User Interface](#user-interface)
5. [Capabilities of Frederick: Answering Student Questions](#capabilities-of-frederick-answering-student-questions)
6. [Conclusion](#conclusion)
7. [References](#references)
8. [Appendix](#appendix)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)
11. [Contact](#contact)
12. [Project Status](#project-status)
13. [Project Structure](#project-structure)
14. [Contributing](#contributing)
15. [Support](#support)
16. [Authors and Acknowledgement](#authors-and-acknowledgement)


---


> <div style="text-align: center;">For Millennials, chatbots are a natural extension of texting. Millennials, having grown up with the Internet and responsive on-demand services, often cite phone calls as being overly intrusive, even presumptuous - thus their preference for texting. <br> <i>— Joseph Vincze (Library Hi Tech News)</i> </div>

<style>
body {
  counter-reset: h2;
}
/*h1::before {*/
/*  counter-increment: h1;*/
/*  content: counter(h1) ". ";*/
/*}*/

h1 {
  counter-reset: h2;
}
h2 {
  counter-reset: h3;
}
h3 {
  counter-reset: h4;
}
h2::before {
  counter-increment: h2;
  content: counter(h2) ". ";
}
h3::before {
  counter-increment: h3;
  content: counter(h2) "." counter(h3) ". ";
}
h4::before {
  counter-increment: h4;
  content: counter(h4) ". ";
}

</style>

## Introduction


The increasing importance of Artificial Intelligence (AI) in various fields has led to the development of AI chatbots, which have revolutionized 
the way humans interact with machines. These conversational agents have become an essential component in many industries, including education. 
The University of Naples Federico II (UNINA) is no exception, as it has the potential to benefit form integrating AI-powered virtual assistants 
into its operations.

One such example is Frederick, a dedicated virtual assistant designed to provide students with accurate and helpful answers to their questions 
about UNINA's courses and programs. This innovative system utilizes a RAG pipeline, leveraging the power of Natural Language Processing (NLP) to 
understand and respond to user queries. The data for this chatbot was carefully scraped from the university website using two methods: a traditional 
approach using Selenium and a more advanced method utilizing Large Language Models (LLMs).

As Frederick's mission is to provide students with accurate and helpful answers, it is essential to develop its capabilities in understanding 
and responding to various types of questions. In this report, we will outline Frederick's role as a virtual assistant and explore the types of 
questions it can answer, including course details such as departments, schools, duration of courses, contact information, and more.


## Background: The Rise of AI Chatbots

The advent of Artificial Intelligence (AI) has revolutionized the way humans interact with machines, giving rise to a new generation of intelligent systems. Among these innovations are AI chatbots, also known as conversational agents or virtual assistants. These software applications use Natural Language Processing (NLP) and machine learning algorithms to understand and respond to human input in the form of text or voice commands.

In recent years, AI chatbots have become increasingly prevalent across various industries, including customer service, healthcare, finance, and more. Their ability to process vast amounts of data, provide rapid responses, and adapt to user behavior has made them an attractive solution for many organizations seeking to improve their communication with customers, employees, or students.

The increasing importance of AI chatbots in modern life is evident in the growing number of companies investing in these technologies. As a result, the development of sophisticated NLP capabilities, advanced machine learning algorithms, and improved user interfaces has become essential to stay ahead in this rapidly evolving landscape.

In this report, we will explore the role of AI chatbots in providing accurate and helpful answers to students' questions about UNINA's courses and programs, as embodied by Frederick, a dedicated virtual assistant designed to serve students at the University of Naples Federico II.


## Frederick: The Virtual Assistant for UNINA Students

As a dedicated virtual assistant, Frederick is designed to provide students with accurate and helpful answers to their questions about UNINA's courses and programs. This system utilizes a RAG pipeline, leveraging the power of Natural Language Processing (NLP) to understand and respond to user queries.

Frederick's primary mission is to serve as a valuable resource for current or prospective students of UNINA, providing them with essential information about course details, coordinator contact information, language of the course, duration, locations, and more. By doing so, Frederick aims to simplify the process of finding relevant information, reducing the time students spend searching for answers online.

To accomplish this mission, we carefully scraped data from the university website using two methods: a traditional approach using Selenium and a more advanced method utilizing Large Language Models (LLMs). This data will serve as the foundation for Frederick's knowledge base, enabling it to provide accurate and helpful responses to student queries.

In the following sections, we will delve into how Frederick is understanding and responding to various types of questions, showcasing its potential to improve the way students interact with UNINA.



## Methodology


### Data Scraping

> <div style="text-align: center;">To the user, chatbots seem to be “intelligent” due to their informative skills. However, chatbots are only as intelligent as the underlying database. <br> <i>— Peter Gentsch (AI in Marketing, Sales and Service)</i></div>

To gather data for Frederick's knowledge base, we employed two methods: traditional scraping using Selenium and advanced scraping utilizing Large Language Models (LLMs). In this section, we will outline the methodology used for both approaches.

**Vanilla Scrape Methodology**

For the traditional scrape method, we utilized Selenium, a popular open-source tool for web scraping. However, we were unable to employ BeautifulSoup due to the UNINA website's use of JavaScript objects to store information. As such, we relied solely on Selenium for data extraction.

To scrape data, we followed these steps:

1. Launched a Selenium WebDriver instance with Chrome browser and enabled headless mode to simulate a user's browsing experience.
2. Visited the UNINA website and used Selenium's `find_element` method to locate specific elements containing desired information using their XPath (e.g., course schedules, instructor contact details).
3. Extracted the data from these elements using Selenium's `text` property or by performing manual string manipulation.
4. Used WebDriverWait and expected_conditions modules to wait for dynamic content to load before scraping the data.
5. Stored the scraped data in a JSON file and a Pandas DataFrame for further processing.

**LLM Scraping Methodology**

To leverage Large Language Models (LLMs) for advanced data scraping, we employed SmartScraperGraph, a powerful open-source tool that utilizes direct graph implementation to retrieve and extract relevant information from web pages.

We used the "gpt-3.5-turbo" model to process queries and generate coherent answers. Specifically, we:

1. Loaded the UNINA website into the SmartScraperGraph scraper.
2. Provided specific query parameters to extract desired information (e.g., course details, coordinator, contact information).
3. The LLM model analyzed the webpage content and returned extracted data in a structured format.
4. Converted the retrieved data to JSON format.
5. Saved the resulting JSON file to an external file for future reference.

By combining these two approaches, we were able to gather a comprehensive dataset for Frederick's knowledge base, enabling it to provide accurate and helpful responses to
student queries.


### RAG Pipeline
<figure>
  <img src="https://i.ibb.co/jzwRBTN/Screenshot-2024-06-18-at-10-12-50.png" alt="RAG-Pipeline" style="width:100%">
  <figcaption style="text-align: center;">RAG Pipeline</figcaption>
</figure>

In this section, we will outline the methodology used for the Retrieval-Augmented-Generation (RAG) pipeline, which enables Frederick to 
provide accurate and helpful responses to student queries.

#### Load the necessary documents

In this step we imported the documents from the datasets prepared by the scraper. Including:
* List of university courses (json, csv)
* Description and necessary information of each course (json, csv)

Also:
* The system message for the chatbot (txt)
* University international welcome-desk information (web)

#### Efficient Document Processing through Segmentation

By breaking down large documents into smaller, more manageable chunks, we can streamline our processing efforts and achieve greater efficiency. This process, known as text segmentation or splitting, involves dividing a massive document into distinct, bite-sized pieces. By doing so, we can analyze and process the text in a more detailed and structured manner, enabling us to extract valuable insights and information from the data.

While segmenting documents may potentially disrupt the original context or meaning of the text, we can employ various techniques to mitigate this issue. For instance, we can utilize recursive splitting methods or incorporate techniques that preserve the original context and meaning of the text.

#### Vector Representation Using Embedding Models

We represent each document as a dense vector using pre-trained embedding models. When converting text into numerical representations, we rely on various embedding models to transform the text into dense vectors that capture its semantic meaning. Popular options include BERT, RoBERTa, GPT-2, among others. In this example, we utilize the MiniLM model, a smaller variant of the BERT architecture. By applying an embedding model, we can convert text into numerical vectors that encode the semantic essence of the text. These vectors can be employed for various natural language processing tasks, such as:

* Text classification: categorizing text based on its content
* Clustering: grouping similar texts together
* Similarity search: identifying the most relevant texts based on their semantic meaning

The embedding model plays a crucial role in enabling these NLP applications by providing a compact and meaningful representation of the original text.

<a href="https://www.elastic.co/what-is/vector-embedding">
<figure>
  <img src="https://images.contentstack.io/v3/assets/bltefdd0b53724fa2ce/blt185ef72de6dc0e43/6466a9a1f21a3540facf75ac/vector-search-diagram-cropped-white-space.png" alt="Vector embeddings pipeline" style="width:100%">
  <figcaption style="text-align: center;">Vector embeddings</figcaption>
</figure>
</a>

#### Efficient Embedding Retrieval Using Caching

When processing new documents, our approach first checks the cache for pre-computed embeddings. If these embeddings are not readily available, the method employs an underlying embedder to generate the embeddings and stores them in the cache for future reference.

By leveraging caching, we can:

* Avoid recalculating embeddings each time, which can be computationally expensive
* Retrieve embeddings from memory, making our process faster and more efficient

To optimize our embedding-based workflow, let's set up a cache to store pre-computed embeddings. This will enable us to quickly retrieve relevant
information and reduce the computational overhead associated with re-calculating embeddings.

#### Vector Database for Efficient Storage and Retrieval

Vector databases are designed to efficiently store and retrieve vectors, making them an essential component in applications such as information retrieval, recommendation systems, and clustering. These databases enable fast similarity searches by comparing query vectors with stored vectors, allowing us to quickly identify the most relevant matches.

In this notebook, we utilize the Chroma vector database, which is optimized for tasks involving similarity search and retrieval.

**Important Note**: Due to the relatively small size of our documents, it's more efficient to pass entire documents to the vector database at once, rather than splitting them. This approach simplifies the process and minimizes potential errors.

After the initial vectorization step, we load pre-computed vectors from cache on subsequent calls, ensuring a seamless and optimized experience for future interactions with the vector database.

<a href="https://www.graft.com/blog/top-open-source-vector-databases">
<figure>
  <img src="https://cdn.prod.website-files.com/640248e1fd70b63c09bd3d09/65ea3ef09d25332356575dee_chroma%20database%20architecture.svg" alt="Chroma is the open-source embedding database" style="width:100%">
  <figcaption style="text-align: center;">Chroma open-source vector embedding database</figcaption>
</figure>
</a>

#### Utilizing the LLM with Groq API

To leverage the power of language models, let's load the Llama model using the Groq API. This will enable us to tap into the capabilities of this
pre-trained model without having to fine-tune it for our specific task. By utilizing the Groq API, we can streamline our workflow and optimize
performance.

In particular, we plan to utilize the `nvidia/Llama3-ChatQA-1.5-8B` model from Hugging Face's model hub. This model has been specifically designed for
chat-style question answering tasks and is expected to provide strong performance in this domain.

#### Evaluating Model Performance on Unseen Questions

We put the model to the test by asking it questions and assessing its ability to provide accurate answers. Since we're working with an unseen dataset, this exercise will help us gauge the model's generalizability and ability to generalize beyond its training data.


| Prompt                                                                                                      | Response                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|:------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Who is the person in charge for autonomous vehicle engineering course in University of Naples Federico II?  | I've searched for the University of Naples Federico II's Autonomous Vehicle Engineering course, but I couldn't find a specific person in charge for the course. The university's website doesn't provide a single point of contact for the course. <br><br> However, I can suggest some alternatives to help you find the contact information: <br><br> 1. **Department of Engineering**: The Autonomous Vehicle Engineering course is likely part of the Department of Engineering at the University of Naples Federico II. You can visit the department's website and look for the contact information of the department's director or the person in charge of the course. <br><br> 2. **Faculty of Engineering**: Another option is to contact the Faculty of Engineering at the University of Naples Federico II. They may be able to provide you with the contact information of the person in charge of the Autonomous Vehicle Engineering course. <br><br> 3. **Contact the University's International Office**: The University of Naples Federico II has an International Office that handles international relations and collaborations. You can contact them and ask for assistance in finding the person in charge of the Autonomous Vehicle Engineering course. <br><br> Here are the contact details for the University of Naples Federico II: <br><li> University of Naples Federico II <br><li> Department of Engineering: [ingegneria.unina.it](http://ingegneria.unina.it) <br> <li> Faculty of Engineering: [ingegneria.unina.it/facolta](http://ingegneria.unina.it/facolta) <br><li> International Office: [internationaloffice.unina.it](http://internationaloffice.unina.it) <br><br> Remember to be respectful and clear in your inquiry, and you may want to provide some background information about your interest in the Autonomous Vehicle Engineering course. Good luck! |


#### Contextual Compression: A Technique for Enhanced Retrieval

**NOTE:** This method is not used in the final implementation.

In this implementation, we utilize contextual compression as a technique to condense the context surrounding a query into a more compact representation. This compressed context is then used to retrieve relevant information from our vector database.

By passing the compressed context to our retrieval model, we enable it to pinpoint the most pertinent documents or vectors with greater accuracy. This approach has the added benefit of reducing the search space, allowing us to focus on the most promising candidates and ultimately improving the overall retrieval process.

#### Crafting Effective Prompt Templates for LLM Models

**NOTE:** Below text was generated by Llama.

To elicit the best possible responses from our LLM models, we'll follow these guidelines to create high-quality prompt templates.

* Ensure that your prompts are concise and easy to understand, avoiding technical jargon and using natural language instead.
* To provide additional context, ask the LLM to role-play as a character with a specific goal. For example, "You are a marketing executive who wants to invite major customers to a live event."
* Refine your prompt templates by iterating on different approaches to achieve the same goal. Seek feedback from end-users and analyze how well your prompts generate desired responses.
* Consistency is key: choose a writing style and stick to it, using consistent word choice, intensifiers, emojis, and punctuation to shape the LLM's
response.
* To help the LLM differentiate between context and instructions, create an "Instructions" section with triple quotes (""") in your prompt template.
This will guide the model to focus on generating specific content rather than discussing the process.
* Include direct instructions for the LLM to generate only the expected type of content. For example, if you want a draft email, specify that the
response should strictly follow these guidelines.

To get started, draw inspiration from the Example Prompt Template Library and customize templates to fit your unique needs. Study the language used
in the examples, paying particular attention to writing style and phrasing. You can apply similar techniques to create effective prompt templates for
your LLM models.

#### Crafting Contextualized Prompt Templates

When formulating prompt templates for our chatbot, we'll consider two key factors: the chat history and the latest user question. This context might
be referenced in the chat history, but we need to create a standalone question that can be understood without it.

Our goal is not to answer the question, but rather to reformulate it if necessary, or return it as is if it's already suitable for our chatbot.

The suggested contextualized system prompt is:
```python
contextualize_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

qa_system_prompt =  SystemMessagePromptTemplate.from_template(
    contextualize_system_prompt
)
```

The contextualized human prompt:
```python
qa_human_prompt_template = """Question: {input}
Answer:"""

qa_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template=qa_human_prompt_template)
)
```

And finally, the contextualized Q prompt will be:
```python
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        qa_system_prompt,
        MessagesPlaceholder("chat_history"),
        qa_human_prompt,
    ]
)
```

#### Establishing a Chain of Thought

In this implementation we use the Chain-of-Thought Prompting technique to guide the LLM model through a series of prompts that build upon each other. This approach helps the model generate more coherent and contextually relevant responses by providing a structured framework for the conversation.


Contextual Retrieval with History Awareness, Sequential Processing and Augmentation such as Document Chain Creation and Retrieval Chain, Session History Storage, and Prompting Techniques such as Contextualization Prompts and QA Prompt Templates are used to enhance the retrieval process.


**Similarity-Based Search**

In the realm of data retrieval from vector databases, the "similarity" search type focuses on identifying documents or vectors that closely match a given query. This approach relies on mathematical metrics like cosine similarity or Euclidean distance to calculate the similarity between the query vector and those stored in the database. Documents with higher similarity scores are deemed more relevant and are returned as search results. The "similarity" search type is commonly employed for tasks such as document retrieval, recommendation systems, and information retrieval.


<a href="https://arxiv.org/abs/2201.11903">
<figure>
  <img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fcot.1933d9fe.png&w=1920&q=75" alt="Standard Prompting vs. Chain-of-Thought Prompting" style="width:100%">
  <figcaption style="text-align: center;">Standard Prompting vs. Chain-of-Thought Prompting</figcaption>
</figure>
</a>



### User Interface
Gradio is a user-friendly tool that enables you to create a web-based interface for your machine learning model. By defining the input and output types of your model, Gradio will automatically generate an interactive web interface that allows users to engage with your model. You can customize the interface by selecting from various input and output types, as well as styling options, to suit your specific needs.

Gradio also provides a shareable link to your interface, making it simple to share your model with others.

When evaluating the Learning Curve of Streamlit and Gradio, a notable distinction arises. Gradio excels in providing a more intuitive interface, making it ideal for users seeking a straightforward and quick development environment. On the other hand, while Streamlit offers advanced customization capabilities, it may require a slightly steeper learning curve for users aiming to harness its full potential.






> <div style="text-align: center">"The contextual knowledge is embedded in one form — the capacity to rattle off linguistic knowledge — but is not embedded in another form" <br> <i>—Yann LeCun</i>

---

**Resources**

* Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2021). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* ([arXiv:2005.11401](http://arxiv.org/abs/2005.11401)). arXiv.

* [ScrapeGraphAI](https://github.com/VinciGit00/Scrapegraph-ai): *A Python library for scraping leveraging large language models*

* Model [*all-MiniLM-L6-v2*](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): Maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

* Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* ([arXiv:1908.10084](http://arxiv.org/abs/1908.10084)). arXiv.

* [A comparison of sentence transformer pretrained_models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models)

* Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers* ([arXiv:2002.10957](http://arxiv.org/abs/2002.10957)). arXiv.

* [Chroma](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/): an AI-native open-source vector database licensed under Apache 2.0 focused on developer productivity and happiness.

* [Meta Llama3 on Groq](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [Llama 3 from Ollama](https://ollama.com/library/llama3): The most capable openly available LLM to date.

* [LangChain Use Cases - Q&A with RAG](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)

* [Galileo: AI chatbot assistant for University of Padova](https://github.com/sinadalvand/Galileo.git)

* Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2023). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models ([arXiv:2201.11903](http://arxiv.org/abs/2201.11903)). arXiv.

* [LangChain in Chains #22: Chain of Thought Prompting](https://ai.plainenglish.io/langchain-in-chains-22-chain-of-thought-prompting-8b0dc4b01215)

---

<div style="align-content: center"><img src="https://fuoridalcoro.altervista.org/wp-content/uploads/2023/11/UNINA-800-LOGO_segnaposto-eventi_620x300.jpg" alt="UNINA Logo 800 Years"></div>
