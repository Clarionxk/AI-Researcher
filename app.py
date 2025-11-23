import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

load_dotenv()

st.set_page_config(page_title="AI Infographic Generator", layout="wide")

# Custom CSS
st.markdown("""
<style>
    
    .info-box {
        background-color: black;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PDF GENERATION ENGINE
# ==========================================

def create_infographic_pdf(data, filename="infographic.pdf"):
    """
    Generates a visual PDF using ReportLab based on structured JSON.
    """
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    # Background
    c.setFillColor(colors.whitesmoke)
    c.rect(0, 0, width, height, fill=1)
    
    # Header
    c.setFillColor(colors.cornflowerblue)
    c.rect(0, height - 1.5*inch, width, 1.5*inch, fill=1, stroke=0)
    
    # Title
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(width/2, height - 1*inch, f"Infographic: {data.get('topic', 'Unknown Topic')}")
    
    # Stats Section
    y_pos = height - 2.5*inch
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.5*inch, y_pos, "Key Statistics")
    
    stats = data.get('stats', [])
    box_width = 2.2*inch
    box_y = y_pos - 1.5*inch
    
    for i, stat in enumerate(stats[:3]): 
        x_pos = 0.5*inch + (i * (box_width + 0.2*inch))
        
        c.setFillColor(colors.white)
        c.roundRect(x_pos, box_y, box_width, 1.2*inch, 10, fill=1, stroke=0)
        
        c.setFillColor(colors.cornflowerblue)
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(x_pos + box_width/2, box_y + 0.7*inch, str(stat.get('value', '')))
        
        c.setFillColor(colors.gray)
        c.setFont("Helvetica", 10)
        c.drawCentredString(x_pos + box_width/2, box_y + 0.4*inch, str(stat.get('label', '')))

    # Summary Section
    y_pos = box_y - 0.5*inch
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.5*inch, y_pos, "Executive Summary")
    
    y_pos -= 0.3*inch
    c.setFont("Helvetica", 11)
    text_object = c.beginText(0.5*inch, y_pos)
    text_object.setFont("Helvetica", 11)
    text_object.setLeading(14)
    
    summary_text = data.get('summary', 'No summary available.')
    words = summary_text.split()
    line = []
    for word in words:
        line.append(word)
        if len(" ".join(line)) > 80: 
            text_object.textLine(" ".join(line))
            line = []
    text_object.textLine(" ".join(line))
    c.drawText(text_object)
    
    # Fun Fact
    y_pos = text_object.getY() - 0.5*inch
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.5*inch, y_pos, "Did You Know?")
    
    y_pos -= 0.3*inch
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(0.5*inch, y_pos, f"• {data.get('fun_fact', '')}")

    c.save()
    return filename

# ==========================================
# 3. CREWAI BACKEND (FIXED)
# ==========================================

def run_crew_research(topic, api_key):
    """
    Orchestrates the CrewAI Agents using the native LLM class.
    """
    
    # --- CRITICAL FIX: Set Env Var for CrewAI Native LLM ---
    # CrewAI's 'LLM' class looks for GOOGLE_API_KEY in os.environ
    os.environ["GOOGLE_API_KEY"] = api_key

    # 1. Define LLM using proper provider prefix 'gemini/'
    gemini_llm = LLM(
        model="gemini/gemini-2.5-flash",
        temperature=0.5,
        api_key=api_key
    )

    # 2. Define Agents
    researcher = Agent(
        role='Senior Research Analyst',
        goal=f'Uncover key statistics and facts about: {topic}',
        backstory="Expert researcher finding precise numbers.",
        llm=gemini_llm,
        verbose=True,
        allow_delegation=False
    )

    designer = Agent(
        role='Information Architect',
        goal='Structure the research into valid JSON',
        backstory="You format raw research into strict JSON for app consumption.",
        llm=gemini_llm,
        verbose=True,
        allow_delegation=False
    )

    # 3. Define Tasks
    research_task = Task(
        description=f"""
        Research '{topic}'. Find:
        1. Executive summary (max 150 words).
        2. 3 numerical stats (e.g., "Market Cap: $1T").
        3. One fun fact.
        """,
        agent=researcher,
        expected_output="Text report with extensive report and summary, stats, and fact."
    )

    design_task = Task(
        description="""
        Format the research into this JSON structure:
        {
            "topic": "Topic Name",
            "summary": "Summary text",
            "stats": [{"label": "L1", "value": "V1"}, {"label": "L2", "value": "V2"}, {"label": "L3", "value": "V3"}],
            "fun_fact": "Fact text"
        }
        Return ONLY valid JSON.
        """,
        agent=designer,
        expected_output="Valid JSON string.",
        context=[research_task]
    )

    # 4. Run Crew
    crew = Crew(
        agents=[researcher, designer],
        tasks=[research_task, design_task],
        verbose=True,
        process=Process.sequential
    )

    return crew.kickoff()

# ==========================================
# 4. STREAMLIT FRONTEND
# ==========================================

def main():
    st.sidebar.title("Configuration")
    
    # Handle API Key
    env_api_key = os.getenv("GOOGLE_API_KEY") or ""
    api_key = st.sidebar.text_input("Gemini API Key", value=env_api_key, type="password")

    st.title("🎨 AI Infographic Generator")
    topic = st.text_input("Enter a topic (e.g., 'SpaceX', 'Bitcoin', 'Climate Change')")

    if st.button("Generate Infographic"):
        if not api_key:
            st.error("Please enter your API Key.")
        elif not topic:
            st.warning("Please enter a topic.")
        else:
            with st.spinner('🤖 Researching & Designing...'):
                try:
                    # Run Crew
                    crew_output = run_crew_research(topic, api_key)
                    
                    # Clean JSON
                    raw_json = str(crew_output).strip()
                    if "```json" in raw_json:
                        raw_json = raw_json.split("```json")[1].split("```")[0]
                    elif "```" in raw_json:
                        raw_json = raw_json.split("```")[1].split("```")[0]
                    
                    data = json.loads(raw_json)

                    # Display
                    st.success("Success!")
                    st.markdown(f"## 📊 {data.get('topic', topic)}")
                    st.info(data.get('summary'))

                    cols = st.columns(3)
                    stats = data.get('stats', [])
                    for i, stat in enumerate(stats):
                        if i < 3:
                            cols[i].metric(label=stat.get('label'), value=stat.get('value'))

                    st.warning(f"Did you know? {data.get('fun_fact')}")

                    # PDF
                    pdf_file = create_infographic_pdf(data)
                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            label="Download PDF",
                            data=f,
                            file_name=f"{topic}_infographic.pdf",
                            mime="application/pdf"
                        )

                except Exception as e:
                    # Error handling fixed: removed reference to 'crew_output' 
                    # because it doesn't exist if the crew failed.
                    st.error(f"Error: {str(e)}")
                    st.warning("If you get a 404 error, check your API Key permissions or try using model='gemini/gemini-1.5-flash-latest' in app.py")

if __name__ == "__main__":
    main()
