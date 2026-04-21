# IMPORTS
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime
import os


# LLM SETUP
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_retries=3
)


# EMBEDDING + CHROMADB
embedder = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="hr_kb")

# DOCUMENTS
doc_001 = {
  "id": "doc_001",
  "topic": "Leave Policy",
  "text": """
The company provides various types of leave to employees to support work-life balance and personal needs. Employees are entitled to annual leave, sick leave, and casual leave as per company policy.

Annual leave is granted at the rate of 18 days per year and must be approved in advance by the reporting manager. Employees are encouraged to plan their leave in advance to ensure proper workload management.

Sick leave allows employees to take time off in case of illness or medical emergencies. A maximum of 10 sick leave days is allowed per year. For absences exceeding two consecutive days, a valid medical certificate must be submitted.

Casual leave is provided for personal reasons or short-term requirements. Employees can avail up to 7 casual leave days annually. These leaves are generally not carried forward to the next year.

Unused annual leave can be carried forward up to a maximum limit of 10 days. Any leave beyond this limit will lapse at the end of the year.

All leave requests must be submitted through the company’s HR portal and approved before the leave is taken, except in emergency situations.
"""
}

doc_002 = {
  "id": "doc_002",
  "topic": "Payroll Policy",
  "text": """
The company follows a structured payroll system to ensure timely and accurate salary disbursement to all employees. Salaries are credited on the last working day of each month directly to the employee’s registered bank account.

The salary structure includes basic pay, house rent allowance (HRA), special allowances, and applicable bonuses. Deductions include provident fund (PF), professional tax, income tax (TDS), and any other statutory deductions.

Employees receive a detailed payslip every month through the HR portal, which outlines earnings, deductions, and net salary.

Any discrepancies in salary must be reported to the HR or payroll team within five working days of salary credit. Corrections, if applicable, will be processed in the next payroll cycle.

Overtime payments, incentives, and bonuses are processed separately and may not always be included in the monthly salary cycle.
"""
}

doc_003 = {
  "id": "doc_003",
  "topic": "Attendance Policy",
  "text": """
Employees are required to adhere to the company’s attendance policy to ensure discipline and productivity. The standard working hours are 9:00 AM to 6:00 PM, Monday to Friday.

Employees must mark their attendance using the company’s biometric system or online attendance portal. Late arrivals beyond 15 minutes may be considered half-day leave if repeated frequently.

A minimum of 9 working hours, including breaks, must be completed daily. Employees are allowed a lunch break of 1 hour.

Absence without prior approval or valid reason may result in salary deductions or disciplinary action. Regular absenteeism is monitored and may impact performance evaluations.

Work-from-home attendance must also be logged through the designated system and approved by the reporting manager.
"""
}

doc_004 = {
  "id": "doc_004",
  "topic": "Insurance Policy",
  "text": """
The company provides health insurance coverage to all full-time employees as part of its benefits package. The insurance policy includes medical coverage for hospitalization, pre- and post-hospitalization expenses, and certain day-care procedures.

Employees are enrolled in the insurance plan from their date of joining. The policy also extends coverage to immediate family members, including spouse and up to two dependent children.

The sum insured is determined based on the employee’s grade and position within the company. Employees can opt for additional coverage by paying an extra premium, which will be deducted from their salary.

Claims can be made through network hospitals using a cashless facility or by reimbursement in non-network hospitals. All claims must be supported by valid medical documents and submitted within the specified time frame.

Details regarding policy coverage, network hospitals, and claim procedures are available on the HR portal or through the HR team.
"""
}

doc_005 = {
  "id": "doc_005",
  "topic": "Employee Benefits",
  "text": """
The company offers a comprehensive range of benefits to support employee well-being and satisfaction. These benefits include health insurance, paid leaves, retirement benefits, and performance-based incentives.

Employees are eligible for provident fund (PF) contributions, where both the employee and employer contribute a fixed percentage of the basic salary. Gratuity benefits are provided to employees who complete a minimum of five years of continuous service.

Additional benefits include flexible working hours, wellness programs, and access to professional development resources such as training programs and certifications.

Performance bonuses and annual increments are based on individual performance and company profitability. Employees may also receive rewards and recognition for exceptional contributions.

All benefits are subject to company policies and may be revised periodically.
"""
}

doc_006 = {
  "id": "doc_006",
  "topic": "Resignation Policy",
  "text": """
Employees who wish to resign from the company must submit a formal resignation through the HR portal or via email to their reporting manager and HR department.

The standard notice period is 30 days from the date of resignation submission. Employees are expected to complete all assigned tasks and ensure proper handover of responsibilities during this period.

In certain cases, employees may request early release, which is subject to approval by management. Alternatively, the company may allow buyout of the notice period as per policy.

Final settlement, including pending salary, leave encashment, and other dues, will be processed after the employee’s last working day and clearance of all company assets.

An exit interview may be conducted to gather feedback and improve organizational practices.
"""
}

doc_007 = {
  "id": "doc_007",
  "topic": "Code of Conduct",
  "text": """
The company expects all employees to maintain a high standard of professional behavior and integrity in the workplace. Employees must adhere to ethical practices, respect colleagues, and comply with company policies.

Harassment, discrimination, or any form of inappropriate behavior is strictly prohibited. Employees are expected to create a safe and inclusive work environment.

Confidential company information must not be disclosed to unauthorized individuals. Employees must use company resources responsibly and only for official purposes.

Any violation of the code of conduct may result in disciplinary action, including warnings, suspension, or termination of employment.

Employees are encouraged to report any misconduct to the HR department without fear of retaliation.
"""
}

doc_008 = {
  "id": "doc_008",
  "topic": "Promotion Policy",
  "text": """
The company follows a structured promotion policy based on employee performance, skills, and organizational requirements. Promotions are typically reviewed annually during performance appraisal cycles.

Employees are evaluated based on key performance indicators (KPIs), contributions to team goals, leadership qualities, and overall impact on the organization.

Recommendations for promotion are made by reporting managers and reviewed by senior management and HR. Final decisions are based on merit and business needs.

Employees are encouraged to continuously upgrade their skills and take on additional responsibilities to be considered for promotion.

Promotion may include a change in job role, responsibilities, and compensation.
"""
}

doc_009 = {
  "id": "doc_009",
  "topic": "Grievance Handling",
  "text": """
The company has a formal grievance handling mechanism to address employee concerns and complaints. Employees can raise grievances related to workplace issues, management decisions, or colleague behavior.

Grievances can be submitted through the HR portal or directly to the HR department. All complaints are treated confidentially and investigated promptly.

The HR team will review the grievance, conduct necessary discussions, and take appropriate action based on findings.

Employees will be informed about the resolution within a reasonable time frame. If not satisfied, employees may escalate the matter to higher management.

The company ensures that no employee faces retaliation for raising genuine concerns.
"""
}

doc_010 = {
  "id": "doc_010",
  "topic": "Work From Home Policy",
  "text": """
The company allows employees to work from home based on role requirements and manager approval. Work-from-home arrangements can be temporary or permanent depending on business needs.

Employees must ensure a productive work environment at home and adhere to standard working hours. Attendance must be marked using the online system.

Regular communication with team members and participation in meetings is mandatory. Employees must be available during working hours and respond to official communications promptly.

Any misuse of the work-from-home policy may lead to revocation of the privilege.

The company may provide necessary support such as remote access tools and IT assistance to ensure smooth functioning.
"""
}

documents = [
        doc_001, doc_002, doc_003, doc_004, doc_005,
        doc_006, doc_007, doc_008, doc_009, doc_010
    ]

# LOAD KNOWLEDGE BASE
def load_knowledge_base():
    doc_texts = [doc["text"] for doc in documents]
    doc_ids = [doc["id"] for doc in documents]
    doc_metadatas = [{"topic": doc["topic"]} for doc in documents]

    embeddings = embedder.encode(doc_texts).tolist()

    if collection.count() == 0:
        collection.add(
            documents=doc_texts,
            embeddings=embeddings,
            ids=doc_ids,
            metadatas=doc_metadatas
        )

load_knowledge_base()

# STATE DEFINITION
class CapstoneState(TypedDict):
    question: str
    messages: List[str]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str


# MEMORY NODE
def memory_node(state: CapstoneState):
    question = state["question"]
    messages = state.get("messages", [])

    messages.append(f"User: {question}")
    messages = messages[-6:]

    user_name = state.get("user_name", "")

    if "my name is" in question.lower():
        user_name = question.lower().split("my name is")[-1].strip().title()

    return {
        "messages": messages,
        "user_name": user_name
    }


# ROUTER NODE
def router_node(state: CapstoneState):
    question = state["question"].lower()

    # Rule-based override (faster + reliable)
    if "date" in question or "time" in question:
        return {"route": "tool"}

    if "my name" in question or "who am i" in question:
        return {"route": "skip"}

    # LLM fallback
    prompt = f"""
        Classify the query into ONE:
        - retrieve (HR policies)
        - tool (date/time)
        - skip (memory/personal)

        Reply ONLY one word.

        Question: {question}
        """

    route = llm.invoke(prompt).content.strip().lower()

    if route not in ["retrieve", "tool", "skip"]:
        route = "retrieve"

    return {"route": route}

# RETRIEVE CONTEXT
def retrieve_context(question: str, k: int = 3):
    query_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    # Format context with topic labels
    context = "\n\n".join(
        f"[{m['topic']}]\n{d}" for d, m in zip(docs, metas)
    )

    sources = [m["topic"] for m in metas]

    return context, sources

# RETRIEVAL NODE
def retrieval_node(state: CapstoneState):
    question = state["question"]

    context, sources = retrieve_context(question)

    return {
        "retrieved": context,
        "sources": sources
    }


# SKIP NODE
def skip_node(state: CapstoneState):
    return {
        "retrieved": "",
        "sources": []
    }


# TOOL NODE
def tool_node(state: CapstoneState):
    try:
        now = datetime.now().strftime("%Y-%m-%d")
        return {"tool_result": now}
    except Exception as e:
        return {"tool_result": str(e)}


# ANSWER NODE
def answer_node(state: CapstoneState):
    question = state["question"]
    context = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages = state.get("messages", [])
    user_name = state.get("user_name", "")
    retries = state.get("eval_retries", 0)

    # -------- MEMORY DIRECT ANSWER --------
    if "my name is" in question.lower():
        return {"answer": f"Nice to meet you, {user_name}."}

    if any(x in question.lower() for x in ["what is my name", "who am i"]):
        if user_name:
            return {"answer": f"Your name is {user_name}."}
        else:
            return {"answer": "I don’t know your name yet."}

    # -------- EMOTIONAL HANDLING --------
    if any(word in question.lower() for word in ["stress", "stressed", "anxious"]):
        return {
            "answer": "I'm sorry you're feeling this way. Please consider reaching out to HR or a professional counselor."
        }

    # -------- MAIN PROMPT --------
    system_prompt = f"""
        You are an HR Policy Assistant.

        STRICT RULES:
        - If Tool Output is NOT empty → answer using ONLY the Tool Output.
        - If Tool Output is empty → answer ONLY from the provided Context.
        - Use the [Topic] sections to locate relevant information in the Context.
        - Answer in a clear, complete sentence (no one-word or very short answers).
        - Do NOT hallucinate or add information not present in Context or Tool Output.
        - If the answer is not found in Context, say exactly: "I don’t know. Please contact HR."
        Context:
        {context}

        Tool Output:
        {tool_result}

        Conversation:
        {messages}

        Retry: {retries}
        """

    response = llm.invoke(system_prompt + "\n\nQuestion: " + question)
    answer = response.content.strip()

    # Avoid duplicate name prefix
    if user_name and not answer.lower().startswith(user_name.lower()):
        answer = f"{user_name}, {answer}"

    return {"answer": answer}


# EVAL NODE
def eval_node(state: CapstoneState):
    context = state.get("retrieved", "")
    answer = state.get("answer", "")
    retries = state.get("eval_retries", 0)

    # Skip eval if no retrieval
    if context == "":
        return {
            "faithfulness": 1.0,
            "eval_retries": retries
        }

    prompt = f"""
        Rate faithfulness (0 to 1):
        Context: {context}
        Answer: {answer}
        Only number.
        """

    score = llm.invoke(prompt).content.strip()

    try:
        score = float(score)
    except:
        score = 0.5

    return {
        "faithfulness": score,
        "eval_retries": retries + 1
    }


# SAVE NODE
def save_node(state: CapstoneState):
    messages = state.get("messages", [])
    answer = state.get("answer", "")

    messages.append(f"Assistant: {answer}")

    return {"messages": messages}


# GRAPH LOGIC
def route_decision(state):
    return state.get("route", "retrieve")


def eval_decision(state):
    if state.get("faithfulness", 1) < 0.7 and state.get("eval_retries", 0) < 2:
        return "answer"
    return "save"


graph = StateGraph(CapstoneState)

graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("skip", skip_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)
graph.add_node("save", save_node)

graph.set_entry_point("memory")

graph.add_edge("memory", "router")

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "retrieve": "retrieve",
        "skip": "skip",
        "tool": "tool"
    }
)

graph.add_edge("retrieve", "answer")
graph.add_edge("skip", "answer")
graph.add_edge("tool", "answer")

graph.add_edge("answer", "eval")

graph.add_conditional_edges(
    "eval",
    eval_decision,
    {
        "answer": "answer",
        "save": "save"
    }
)

graph.add_edge("save", END)


# FINAL APP
app = graph.compile(checkpointer=MemorySaver())