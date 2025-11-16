import streamlit as st
import os

# Fallback built-in questions (minimal set)
BUILTIN_QUESTIONS = [
    {
        "question": "What does KDD stand for?",
        "options": ["Key Data Distribution", "Knowledge Discovery in Databases", "Kernel Data Design", "Known Data Determination"],
        "answer": "B"
    }
]

def load_questions_from_file(filename="questions.txt"):
    if not os.path.exists(filename):
        return None
    questions = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        i = 0
        while i < len(lines):
            question = lines[i]
            options = []
            for j in range(1, 5):
                if i + j < len(lines):
                    opt = lines[i + j]
                    if any(opt.startswith(prefix) for prefix in ["A.", "B.", "C.", "D."]):
                        options.append(opt[3:].strip())
                    else:
                        raise ValueError(f"Line {i+j+1}: Expected 'A. ...'")
                else:
                    raise ValueError("Incomplete question block")
            if i + 5 < len(lines) and lines[i + 5].startswith("Answer:"):
                ans = lines[i + 5].split(":", 1)[1].strip().upper()
                if ans in ["A", "B", "C", "D"]:
                    questions.append({"question": question, "options": options, "answer": ans})
                else:
                    raise ValueError(f"Invalid answer: {ans}")
            else:
                raise ValueError("Missing Answer line")
            i += 6
        return questions
    except Exception as e:
        st.warning(f"⚠️ Error loading questions: {e}")
        return None

def main():
    st.set_page_config(page_title="🧠 DM & ML Quiz", layout="centered")
    st.title("🧠 Data Mining & Machine Learning Quiz")
    
    questions = load_questions_from_file("questions.txt")
    if questions is None:
        questions = BUILTIN_QUESTIONS
    
    if not questions:
        st.error("❌ No questions loaded.")
        return

    total = len(questions)
    
    if "current" not in st.session_state:
        st.session_state.current = 0
        st.session_state.score = 0
        st.session_state.submitted = False

    idx = st.session_state.current
    q = questions[idx]

    st.progress((idx + 1) / total)
    st.subheader(f"Question {idx + 1} of {total}")
    st.write(f"**{q['question']}**")

    user_ans = st.radio("Select your answer:", 
                       [f"A. {q['options'][0]}", 
                        f"B. {q['options'][1]}", 
                        f"C. {q['options'][2]}", 
                        f"D. {q['options'][3]}"],
                       index=None,
                       key=f"q{idx}")

    if st.button("✅ Submit"):
        if user_ans is None:
            st.warning("Please select an answer.")
        else:
            st.session_state.submitted = True
            selected_letter = user_ans[0]  # Extract 'A', 'B', etc.
            if selected_letter == q["answer"]:
                st.success("✅ Correct!")
                st.session_state.score += 1
            else:
                st.error(f"❌ Incorrect. The correct answer is **{q['answer']}. {q['options'][ord(q['answer']) - ord('A')]}**")

    if st.session_state.submitted:
        if idx < total - 1:
            if st.button("➡️ Next Question"):
                st.session_state.current += 1
                st.session_state.submitted = False
                st.rerun()
        else:
            st.divider()
            final = st.session_state.score
            st.title("🎉 Quiz Completed!")
            st.subheader(f"Your Score: **{final} / {total}**")
            pct = (final / total) * 100
            st.metric("Accuracy", f"{pct:.1f}%")
            if pct >= 80:
                st.balloons()
                st.success("Excellent! 🌟")
            elif pct >= 60:
                st.info("Good job! Keep studying. 📚")
            else:
                st.warning("Review the material and try again! 💪")

if __name__ == "__main__":
    main()
