import streamlit as st
import os

# ================================
# BUILT-IN QUESTIONS (fallback)
# ================================
BUILTIN_QUESTIONS = [
    {
        "question": "What does KDD stand for in data mining?",
        "options": ["Key Data Distribution", "Knowledge Discovery in Databases", "Kernel Data Design", "Known Data Determination"],
        "answer": "B"
    },
    {
        "question": "Which of the following is NOT a type of machine learning?",
        "options": ["Supervised learning", "Unsupervised learning", "Reinforcement learning", "Deterministic learning"],
        "answer": "D"
    },
    {
        "question": "According to the lecture notes, what percentage of effort is typically spent on data cleaning and pre-processing?",
        "options": ["20%", "40%", "60%", "80%"],
        "answer": "C"
    }
    # Add more if needed, but we'll primarily use questions.txt
]

# ================================
# LOAD QUESTIONS FROM FILE
# ================================
def load_questions_from_file(filename="questions.txt"):
    if not os.path.exists(filename):
        return None
    questions = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        i = 0
        while i < len(lines):
            question_text = lines[i]
            options = []
            for j in range(1, 5):
                if i + j < len(lines):
                    opt_line = lines[i + j]
                    if opt_line.startswith(("A.", "B.", "C.", "D.")):
                        options.append(opt_line[3:].strip())
                    else:
                        raise ValueError("Option line must start with A., B., C., or D.")
                else:
                    raise ValueError("Incomplete question block")
            
            if i + 5 < len(lines) and lines[i + 5].startswith("Answer:"):
                ans = lines[i + 5].split(":", 1)[1].strip().upper()
                if ans in ["A", "B", "C", "D"]:
                    questions.append({
                        "question": question_text,
                        "options": options,
                        "answer": ans
                    })
                else:
                    raise ValueError(f"Invalid answer: {ans}")
            else:
                raise ValueError("Missing or invalid Answer line")
            
            i += 6
        return questions
    except Exception as e:
        st.warning(f"⚠️ Error reading {filename}: {e}. Using built-in questions.")
        return None

# ================================
# MAIN APP
# ================================
def main():
    st.set_page_config(page_title="🧠 DM & ML Quiz", layout="centered")
    st.title("🧠 Data Mining & Machine Learning Quiz")
    
    # Load questions
    questions = load_questions_from_file("questions.txt")
    if questions is None:
        questions = BUILTIN_QUESTIONS
    
    if not questions:
        st.error("❌ No questions available. Please check your `questions.txt` or built-in list.")
        return

    total_questions = len(questions)

    # Initialize session state
    if "current_q" not in st.session_state:
        st.session_state.current_q = 0
        st.session_state.score = 0
        st.session_state.submitted = False
        st.session_state.user_answers = [None] * total_questions

    idx = st.session_state.current_q
    q = questions[idx]

    # Display progress
    st.progress((idx + 1) / total_questions)
    st.subheader(f"Question {idx + 1} of {total_questions}")

    # Display question
    st.markdown(f"**{q['question']}**")

    # Radio buttons for answer
    user_answer = st.radio(
        "Select your answer:",
        ["A", "B", "C", "D"],
        index=None,
        key=f"q{idx}"
    )

    # Submit button
    if st.button("✅ Submit"):
        if user_answer is None:
            st.warning("Please select an answer.")
        else:
            st.session_state.submitted = True
            st.session_state.user_answers[idx] = user_answer
            if user_answer == q["answer"]:
                st.success("✅ Correct!")
                st.session_state.score += 1
            else:
                st.error(f"❌ Incorrect. The correct answer is **{q['answer']}**.")

    # Next button (only after submission)
    if st.session_state.submitted:
        if idx < total_questions - 1:
            if st.button("➡️ Next Question"):
                st.session_state.current_q += 1
                st.session_state.submitted = False
                st.rerun()
        else:
            # Final score
            st.divider()
            final_score = st.session_state.score
            st.title("🎉 Quiz Completed!")
            st.subheader(f"Your Final Score: **{final_score} / {total_questions}**")
            percentage = (final_score / total_questions) * 100
            st.metric(label="Accuracy", value=f"{percentage:.1f}%")
            
            if percentage >= 80:
                st.balloons()
                st.success("Excellent work! 🌟")
            elif percentage >= 60:
                st.info("Good job! Keep studying! 📚")
            else:
                st.warning("Review the material and try again! 💪")

if __name__ == "__main__":
    main()
