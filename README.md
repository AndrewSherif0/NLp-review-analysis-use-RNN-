
# Download latest version
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

print("Path to dataset files:", path)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# تحميل البيانات
df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

# تحويل التصنيفات إلى أرقام (1 للمراجعات الإيجابية، 0 للسلبية)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# تعريف المتغيرات المهمة
vocab_size = 20000  # عدد الكلمات الفريدة في القاموس
max_length = 300    # الطول الأقصى للمراجعات

# تجهيز التوكنز
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])

# تحويل النصوص إلى أرقام
sequences = tokenizer.texts_to_sequences(df['review'])
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
y = df['sentiment'].values

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data Prepared Successfully!")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# إعداد المتغيرات
vocab_size = 20000  # عدد الكلمات الفريدة في القاموس
max_length = 300    # الطول الأقصى لكل مراجعة

# بناء النموذج
# بناء النموذج
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),  # طبقة التضمين
    LSTM(128, return_sequences=False, dropout=0.1, recurrent_dropout=0.1),  # استبدلنا RNN بـ LSTM
    Dense(64, activation='relu'),
    Dropout(0.1),  
    Dense(1, activation='sigmoid')  # إخراج ثنائي
])

# تجميع النموذج
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
# بناء النموذج قبل عرض الملخص
model.build(input_shape=(None, max_length))

# عرض ملخص النموذج مرة واحدة فقط
model.summary()




# بيانات تجريبية (عينة عشوائية بحجم واحد)
dummy_input = np.random.randint(0, vocab_size, (1, max_length))

# تمرير البيانات عبر النموذج
model.predict(dummy_input)
# تدريب النموذج
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
# تقييم الأداء على بيانات الاختبار
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
# تقييم النموذج على بيانات الاختبار
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"🔥 Test Accuracy: {test_acc:.4f}")
print(f"🎯 Test Loss: {test_loss:.4f}")

# تجربة نموذج بعينة جديدة من البيانات
sample_text = ["This movie was fantastic! I really loved the story and the acting."]
sample_sequence = tokenizer.texts_to_sequences(sample_text)
sample_padded = pad_sequences(sample_sequence, maxlen=max_length, padding='post', truncating='post')

# توقع المراجعة الجديدة
prediction = model.predict(sample_padded)
sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
print(f"🔍 Sample Prediction: {sentiment} ({prediction[0][0]:.4f})")

