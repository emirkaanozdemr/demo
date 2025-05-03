import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

model=load_model("model.h5")
st.title("Yaprak Hastalığını Tahmin Et")
img=st.camera_input("Kamera")
def process_image(input_img):
    if input_img.mode == 'RGBA':
        input_img = input_img.convert('RGB')
    input_img=input_img.resize((110,110)) 
    input_img=np.array(input_img)
    input_img=input_img/255.0
    input_img=np.expand_dims(input_img,axis=0)
    return input_img
if st.button("Tahmin Et") and img is not None:
    img=Image.open(img)
    image=process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)
    class_names = ["Black Rot: Black Rot hastalığı, özellikle üzüm bağlarında görülen ve Xanthomonas campestris pv. viticola adlı bakteri tarafından oluşturulan ciddi bir bitki hastalığıdır. Bu hastalık, yapraklarda sararma, kahverengileşme ve siyah lezyonlarla kendini gösterir; zamanla meyvelerde de çürüme ve siyahlaşma meydana gelir. Nemli ve sıcak hava koşulları hastalığın yayılmasını kolaylaştırır. Enfekte olmuş bitkiler verim kaybına uğrayabilir, hatta tamamen kuruyabilir. Black Rot hastalığına karşı etkili mücadele yöntemleri arasında hastalıklı bitki artıklarının temizlenmesi, dayanıklı çeşitlerin tercih edilmesi ve uygun tarım ilaçlarının kullanılması yer alır.", "ESCA: ESCA hastalığı, genellikle yaşlı asmalarda görülen, odun dokularını tahrip eden ve üzüm bağlarında ciddi ekonomik kayıplara yol açan karmaşık bir mantar hastalığıdır. Hastalık etmenleri arasında Phaeomoniella chlamydospora, Phaeoacremonium minimum ve Fomitiporia mediterranea gibi farklı mantar türleri bulunur. ESCA, yapraklarda damar arası sararmalar, kahverengileşmeler, yanıklık benzeri lekeler ve odun dokusunda siyah çizgilenmeler şeklinde belirti verir. İleri safhalarda asmalar aniden solabilir ve tamamen kuruyabilir. Kimyasal mücadelesi sınırlı olan ESCA hastalığına karşı en etkili yöntemler arasında hastalıklı odun kısımlarının uzaklaştırılması, budama aletlerinin dezenfekte edilmesi ve sağlıklı fidan kullanımı yer alır.", "Sağlıklı: Yaprakta hastalık yok.", "Leaf Blight: Leaf Blight (Yaprak Yanıklığı), birçok tarım bitkisinde görülebilen, yaprakların kurumasına ve dökülmesine neden olan yaygın bir bitki hastalığıdır. Genellikle Alternaria, Helminthosporium veya Phytophthora gibi mantar türleri ya da bazı bakteriler tarafından oluşturulur. Hastalık, yapraklarda önce küçük sarı veya kahverengi lekeler olarak başlar, zamanla bu lekeler büyür, birleşir ve yaprakların büyük bir kısmını kaplayarak yanık görünümüne sebep olur. Nemli ve sıcak hava koşulları hastalığın yayılımını hızlandırır. Leaf Blight, fotosentezi olumsuz etkileyerek bitkinin gelişimini ve ürün verimini ciddi şekilde düşürebilir. Mücadele yöntemleri arasında uygun ilaçlama, dayanıklı çeşitlerin tercih edilmesi ve tarla hijyenine dikkat edilmesi yer alır."]
    st.write(class_names[predicted_class])
