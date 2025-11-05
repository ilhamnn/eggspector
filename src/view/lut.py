import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from pathlib import Path
import numpy as np
from PIL import Image
import io
import base64

st.set_page_config(page_title="EggSpector", page_icon="🦖")
st.title("EggSpector")


# CSS dengan base64
def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()

        style = f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpeg;base64,{b64_encoded});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                z-index: 1;
            }}
            .content {{
                position: relative;
                z-index: 2;
                padding: 0px;
                margin: 0px;
            }}
            [data-testid="stVerticalBlockBorderWrapper"] {{
                background-color: rgba(0, 0, 0, 0.5) !important;
                border: none !important;
                padding: 15px !important;
                width: 100% !important;
                border-radius: 10px !important;
                margin-bottom: 5px !important;
                backdrop-filter: blur(5px) !important;
            }}
            [class="st-emotion-cache-10trblm e1nzilvr1"] {{
                justify-content: center !important;
                text-align: center !important;
                width: 100% !important;
            }}
            </style>
        """
        st.markdown(style, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"❌ File tidak ditemukan: {image_file}")
        st.info("Pastikan path file sudah benar!")
    except Exception as e:
        st.error(f"❌ Error saat load background: {str(e)}")


set_background("pict/bg.jpg")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Fungsi untuk prediksi gambar
def predict_image(img_array, model_path):
    class_names = ["Telur Ayam", "Telur Bebek"]

    try:
        model = tf.keras.models.load_model(model_path)
        output = model.predict(img_array)
        score = tf.nn.softmax(output[0])
        confidence = np.max(score)

        if confidence < 0.40:
            return None, confidence

        predicted_class = class_names[np.argmax(score)]
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Terjadi kesalahan saat pemrosesan gambar/prediksi: {e}")
        return None, None


def resize_image(image_path, width, height):
    img = Image.open(image_path)
    img = img.resize((width, height))
    return img


# NAVIGATION in Sidebar
st.sidebar.title("Navigation")

# Radio untuk navigasi
navigation = st.sidebar.radio("", ["Spector", "About", "Type"])


if navigation == "Spector":
    option = st.selectbox(
        label="Pilih Model",
        options=("VGG16", "MobileNetV2"),
        index=None,
        placeholder="Pilih Metode Yang Akan Digunakan...",
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        upload = st.file_uploader(
            "Unggah gambar telur untuk mendapatkan hasil prediksi",
            type=["jpg", "jpeg", "png"],
        )
    with col2:
        st.subheader("Hasil prediksi:")
    with col1:
        if st.button("Predict", type="primary"):
            if upload is not None:
                try:
                    with st.spinner("Memproses gambar untuk prediksi..."):
                        if option == "VGG16":
                            model_path = Path("model/vgg16.keras")
                        elif option == "MobileNetV2":
                            model_path = Path("model/mnv2.keras")
                        else:
                            st.error("Model tidak valid!")
                            st.stop()

                        image_bytes = upload.getvalue()
                        img_array = preprocess_image(image_bytes)
                        result, confidence = predict_image(img_array, model_path)

                    with col2:
                        img = Image.open(upload)
                        img = img.resize((300, 200))
                        st.image(img, caption="Gambar yang diunggah")
                        if result is None:
                            st.error(
                                "⚠️ Gambar yang diunggah tidak termasuk dalam kategori yang dikenali!"
                            )
                            if confidence is not None:
                                with col1:
                                    st.warning(
                                        f"""
                                        Tingkat keyakinan model: {confidence:.2%}

                                        Model saat ini hanya dapat mendeteksi:
                                        - Telur Ayam
                                        - Telur Bebek
                                    """
                                    )
                        else:
                            st.success(f"Hasil Prediksi: **{result}**")
                            st.write(f"Tingkat Kepercayaan: **{confidence:.2%}**")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
            else:
                st.warning("Unggah gambar terlebih dahulu!!")

elif navigation == "About":
    st.header("Hey, Selamat Datang di EggSpector! 🕵️🥚")
    st.write(
        """
        **EggSpector** hadir untuk membantu kamu, peternak dan pengusaha telur, 
        membedakan telur **ayam** dan **bebek** dengan cepat dan mudah! 🚀  
        Kita tahu, kadang susah bedain telur cuma dari cangkangnya, tapi tenang, 
        EggSpector punya **mata AI super tajam** buat bantuin kamu! 👀✨  

        **Apa aja yang bisa EggSpector lakukan?**  
        - Deteksi otomatis:  
        🐓 **Telur Ayam**  
        🦆 **Telur Bebek**  
        - Super gampang: tinggal upload foto telur, dan BOOM—jawabannya langsung ada!  
        - Akurasi tinggi karena kita pakai teknologi deep learning terkini. 💡  

        Cobain sekarang dan lihat betapa kerennya si EggSpector ini!  
        """
    )

elif navigation == "Type":
    col1, col2 = st.columns(2)

    width = 300
    height = 200

    with col1:
        img = resize_image("pict/telurayam.jpg", width, height)
        st.image(img, caption="Telur Ayam")
        st.markdown(
            """
        **Penjelasan:**  
        Telur ayam adalah jenis telur yang sering digunakan untuk kebutuhan sehari-hari dan merupakan sumber protein yang populer.  

        **Ciri-ciri:**  
        - Ukuran biasanya lebih kecil dibandingkan telur bebek.  
        - Cangkang berwarna putih atau cokelat muda dengan tekstur halus.  
        - Isi telur memiliki rasa yang ringan dan cocok untuk berbagai jenis masakan.  
        - Lebih mudah ditemukan di pasaran dibandingkan telur bebek.
        """
        )

    with col2:
        img = resize_image("pict/telurbebek.jpg", width, height)
        st.image(img, caption="Telur Bebek")
        st.markdown(
            """
        **Penjelasan:**  
        Telur bebek biasanya digunakan untuk hidangan khas atau dibuat menjadi telur asin karena rasanya yang lebih kuat.  

        **Ciri-ciri:**  
        - Ukuran lebih besar dibandingkan telur ayam.  
        - Cangkang berwarna putih atau abu-abu dengan tekstur kasar.  
        - Isi telur lebih berminyak dengan rasa yang lebih kaya.  
        - Biasanya digunakan untuk masakan khas seperti martabak atau telur asin.
        """
        )
