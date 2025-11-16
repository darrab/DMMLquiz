import streamlit as st

# Load questions (same function as your console app)
questions = load_questions_from_file() or BUILTIN_QUESTIONS

st.title("🧠 Data Mining Quiz")

if 'current_q' not in st.session_state:
    st.session_state.current_q = 0
    st.session_state.score = 0
    st.session_state.answered = False

q = questions[st.session_state.current_q]
st.subheader(f"Question {st.session_state.current_q + 1}/{len(questions)}")
st.write(q["question"])

# Radio button for answer
user_answer = st.radio("Choose:", ["A", "B", "C", "D"], key="radio")

# Submit button
if st.button("Submit"):
    st.session_state.answered = True
    if user_answer == q["answer"]:
        st.success("✅ Correct!")
        st.session_state.score += 1
    else:
        st.error(f"❌ Incorrect. Correct answer: {q['answer']}")

# Next button
if st.session_state.answered and st.session_state.current_q < len(questions) - 1:
    if st.button("Next"):
        st.session_state.current_q += 1
        st.session_state.answered = False
        st.experimental_rerun()

# Show final score
if st.session_state.current_q == len(questions) - 1 and st.session_state.answered:
    st.title(f"🎉 Final Score: {st.session_state.score}/{len(questions)}")