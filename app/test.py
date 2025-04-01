import os
from dotenv import load_dotenv
import pandas as pd
from math import radians, cos, sin, sqrt, atan2
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import pinecone

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

class SimpleTextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    def load(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [Document(page_content=text, metadata={})]

class ChatBot:
    def __init__(self):
        load_dotenv()
        
        # Load route data from CSV
        try:
            self.routes_df = pd.read_csv('https://dagshub.com/Omdena/VITBhopalUniversity_ChatbotforBRTSNavigation/raw/99c2e8d2883dd9faaa68ed60d5405dd40e77c456/src/tasks/task-2/Routes/all_routes_combined.csv')
        except Exception as e:
            print(f"Error loading routes CSV: {e}")
            return
        
        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Initialize Pinecone
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            print("Pinecone API key missing. Check your .env file.")
            return
        pinecone.init(api_key=api_key, environment='gcp-starter')  
        index_name = "langchain-demo"
        
        # Create the index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, metric="cosine", dimension=768)
                
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.loader = SimpleTextLoader('./routes.txt') 
        self.documents = self.loader.load()
        docs = text_splitter.split_documents(self.documents)
                
        # Batch processing for adding documents
        self.docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        self.vectorstore = self.docsearch
        
        # Use a suitable model for route assistance
        huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not huggingface_api_key:
            print("Hugging Face API key missing. Check your .env file.")
            return
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.7,
            top_k=30,
            max_new_tokens=128,
            huggingfacehub_api_token=huggingface_api_key,
            model_kwargs={}
        )

        # Route assistant prompt with context for guidance
        template = """
        You are a route assistant for the BRTS system. Users will ask you questions about routes, stops, and how to get to their destinations. 
        Use the following route information to assist them. If you don't know the answer, just say you don't know. 
        Context: {context}
        Question: {question}
        Answer: 
        """
        self.prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )
        self.rag_chain = (
            {"context": self.vectorstore.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt 
            | self.llm
            | StrOutputParser()
        )
    
    def haversine(self, lat1, lon1, lat2, lon2):
        # Haversine formula to calculate distance between two points on Earth
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371  # Radius of Earth in kilometers
        return c * r

    def find_closest_bus_station(self, current_lat, current_lon):
        # Calculate distance for each bus station and find the closest
        self.routes_df['distance'] = self.routes_df.apply(
            lambda row: self.haversine(current_lat, current_lon, row['Latitude'], row['Longitude']),
            axis=1
        )
        closest_station = self.routes_df.loc[self.routes_df['distance'].idxmin()]
        return closest_station

    def ask(self, user_input):
        try:
            if "navigate" in user_input.lower():
                parts = user_input.split(" ")
                if len(parts) < 2 or "," not in parts[1]:
                    return "Please provide coordinates in the format: 'navigate latitude,longitude'"
                
                lat_lon = parts[1].split(",")
                current_lat, current_lon = float(lat_lon[0]), float(lat_lon[1])
                closest_station = self.find_closest_bus_station(current_lat, current_lon)
                return (
                    f"Here is the nearest bus stop based on your location.\n\n"
                    f"**Closest Stop:** {closest_station['Station']}\n"
                    f"**Location:** {closest_station['Latitude']}, {closest_station['Longitude']}\n"
                    f"[Navigate using Google Maps](https://www.google.com/maps/dir/{current_lat},{current_lon}/{closest_station['Latitude']},{closest_station['Longitude']})"
                )
            else:
                result = self.rag_chain.invoke(user_input) 
                return result
        except ValueError as e:
            return f"Invalid input: {e}. Please ensure coordinates are in 'latitude,longitude' format."
        except Exception as e:
            print(f"Error during invocation: {e}")
            return "Sorry, I couldn't process your request."

if __name__ == "__main__":
    bot = ChatBot()
    user_input = input("Ask me anything: ")
    result = bot.ask(user_input)
    print(result)
