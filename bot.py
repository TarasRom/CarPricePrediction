from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# 1 Load documents from directory
loader = DirectoryLoader(
    "F:/Download/Manual/",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# 2 Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=300)
chunks = text_splitter.split_documents(documents)

# 3 Create embeddings and a vector database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.from_documents(chunks, embeddings)

# 4 Create the RetrievalQA chain
retriever = db.as_retriever(search_kwargs={"k": 5})
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    chain_type="map_reduce",
    retriever=retriever
)

# 5 Define bot functions

# Handler for the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Привіт! Я бот сервісної служби. Задайте мені питання про ваш принтер.")

# Handler for text messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    answer = qa.invoke(query)
    answer_text = answer["result"]
    await update.message.reply_text(f"🖨️ {answer_text}")

# 6 Initialize and run the Telegram bot
BOT_TOKEN = "YOUR_BOT_TOKEN"
app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Run the bot
print("✅ Бот запущений. Очікує повідомлення...")
app.run_polling()


