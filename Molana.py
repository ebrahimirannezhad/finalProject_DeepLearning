import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import LambdaCallback

# مسیر فایل حاوی اشعار مولانا
file_path = '/content/moulavi_norm.txt'

# خواندن متن از فایل
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read().lower()

# بدست آوردن یک لغت‌نامه برای تبدیل حروف به اعداد و بالعکس
chars = sorted(list(set(text)))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

# طول بلوک‌های متنی برای آموزش مدل
seq_length = 100

# آماده‌سازی داده‌ها
data_X = []
data_y = []
for i in range(len(text) - seq_length):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    data_X.append([char_to_int[char] for char in seq_in])
    data_y.append(char_to_int[seq_out])

X = np.reshape(data_X, (len(data_X), seq_length, 1))
X = X / float(len(chars))
y = np_utils.to_categorical(data_y)

# تعریف مدل LSTM
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# تعریف تابعی برای تولید شعر در طول آموزش مدل
def generate_poetry(epoch, _):
    start_index = np.random.randint(0, len(data_X) - 1)
    pattern = data_X[start_index]
    output = ""
    for i in range(100):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(chars))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        output += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print(output)

# تعریف یک callback برای چاپ شعرهای تولید شده به هنگام آموزش
poetry_generator = LambdaCallback(on_epoch_end=generate_poetry)

# آموزش مدل
model.fit(X, y, epochs=100, batch_size=128, callbacks=[poetry_generator])
# تعداد حروف مورد نظر برای تولید شعر
num_chars_to_generate = 200

# انتخاب یک الفبای تصادفی به عنوان الفبای اولیه برای تولید شعر
start_index = np.random.randint(0, len(data_X) - seq_length)
pattern = data_X[start_index]
generated_poetry = ""

# تولید شعر
for i in range(num_chars_to_generate):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(chars))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    generated_poetry += result
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("شعر تولید شده:")
print(generated_poetry)