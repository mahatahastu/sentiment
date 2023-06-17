import streamlit as st
import joblib

import os






def main():
    
    st.title("Sentiment Analysis")
    # Read the CSV file if it exists
    svm_path = "svm_model.pkl"
    tfidf_path = "tfidf_vectorizer.pkl"
    
    # Check if 'stemmed.csv' file exists
    if not os.path.exists(svm_path):
        st.warning("Proses pembobotan dulu")
    else:
        # Load the SVM model and TF-IDF vectorizer
        svm_model = joblib.load("svm_model.pkl")
        
        # Check if 'stemmed.csv' file exists
        if not os.path.exists(tfidf_path):
            st.warning("Proses pembobotan dulu")
        else:
            # Load the SVM model and TF-IDF vectorizer
            vectorizer = joblib.load("tfidf_vectorizer.pkl")


            
            # Menampilkan hasil evaluasi
            new_string_vector = vectorizer.transform([st.text_area("Masukkan teks:", value="")])


            if st.button("Prediksi"):
                # Melakukan prediksi sentimen
                prediction = svm_model.predict(new_string_vector)
                
                
                if prediction == "positif":
                    st.success("Positif")
                    
                elif prediction == "netral":
                    st.info("Netral")
                
                elif prediction == "negatif":
                    st.error("Negatif")

                    
                #col1, col2 = st.columns(2)
                #with col1:
                #    st.info("Result")

                        
                        
                #with col2:
                #    st.info("Token Sentiment")


if __name__ == '__main__':
    main()