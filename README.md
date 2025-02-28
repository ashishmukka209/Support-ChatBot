CDP Chatbot - AI Support Agent
"Instructions & Setup"
Clone this repository
  git clone https://github.com/ashishmukka209/cdp-chatbot.git
  cd cdp-chatbot
Install the dependencies
  pip install requests beautifulsoup4 langchain chromadb tiktoken sentence-transformers
Set Hugging Face token
  set HUGGINGFACEHUB_API_TOKEN=your_token_here  #Enter your hugging face token 
To run the project
  python chat.py

Working of the Project
Collecting Information 
  The chatbot visits official documentation websites(Segment, mParticle, Lytics, Zeotap) and extracts useful content using web scraping. This allows it to gather the necessary information to help users with their questions.  

Understanding the Information  
  Instead of storing plain text, the chatbot processes the extracted content into a structured format using Hugging Face’s `all-MiniLM-L6-v2` embeddings. This helps it understand the relationships between words and concepts, making search results more accurate.  

Storing for Fast Search  
  The processed data is stored in a database called ChromaDB, which allows the chatbot to quickly retrieve relevant information without having to search through the entire dataset every time a user asks a question.  

Finding & Generating Answers
When a user asks a question, the chatbot:  
- Finds the most relevant content from its stored database.  
- Uses a Hugging Face language model (such as `flan-t5-large`) to generate a clear and useful response based on the retrieved information.  

This approach ensures that the chatbot can provide accurate and relevant answers to CDP-related "how-to" questions. 
