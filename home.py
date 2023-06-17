import streamlit as st
import os

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="👋",
)

st.title("Home")
st.sidebar.success("Pilih Halaman diatas")

# Streamlit app code
def main():
    st.markdown("catatan :")
    st.markdown("Pada sistem sentiment analysis ini data menggunakan data file dengan format csv, jangan lupa menambahkan label (positif, netral, dan negatif) dan jangan lupa data teks kolom di ganti (full_text)")

    col1, col2, col3 = st.columns(3)
    with col1:
    # Delete "stemmed_data.csv"
        if st.button("Delete stemmed_data.csv"):
            delete_file("stemmed_data.csv")
            st.success("stemmed_data.csv deleted successfully.")
    with col2:
        # Delete "svm_model.pkl"
        if st.button("Delete svm_model.pkl"):
            delete_file("svm_model.pkl")
            st.success("svm_model.pkl deleted successfully.")
    with col3:
        # Delete "tfidf_vectorizer.pkl"
        if st.button("Delete tfidf_vectorizer.pkl"):
            delete_file("tfidf_vectorizer.pkl")
            st.success("tfidf_vectorizer.pkl deleted successfully.")

# Function to delete a file
def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        st.error(f"Error deleting {file_path}: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()