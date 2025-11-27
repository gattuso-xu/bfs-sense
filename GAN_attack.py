import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from insert import generate_mask, insert , resample_bfi
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

# 前奏1，加载 TensorFlow 模型
perception_system_tf = load_model('./target_model.h5')

# 前奏2，加载数据集
def load_dataset(file_path):
    data = scio.loadmat(file_path)
    dataset = data['out']  # 假设数据集存储在 'out' 键中
    dataset = np.abs(dataset)  # 取模，转换为实数
    return dataset

#前奏3，任意选择一个被攻击的对象,也就是原始A（这里选择circle.mat）
mat_file_path = './data/circle.mat'  # 替换为你的 .mat 文件路径
mat_data = scio.loadmat(mat_file_path)
matrix = mat_data['out']  # 替换为实际的变量名
#print("检查矩阵的形状:", matrix.shape)
# 随机选择第一维度中的一个索引
random_index = np.random.choice(matrix.shape[0])
# 提取对应的子矩阵,此时obj_A的大小是S*4*234,目前的S是25
orig_A = matrix[random_index]

#(1)A->A*的resample过程
#获取时间戳数组，长度也是 S 。每个时间戳对应一个 4×234 的值集合。时间戳单位：秒
S = orig_A.shape[0]
timestamp = np.arange(0, S * 0.05, 0.05)
#A*大小为112*4*234
A_new = resample_bfi(orig_A, timestamp, target_freq=40)

#（2）GAN的伪造过程：A*->fake_A
# 定义生成器
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(25 * 4 * 234, activation='tanh'))
    model.add(Reshape((25, 4, 234)))
    return model

# 定义判别器
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(25, 4, 234)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

#定义指标一：误分类概率
#使得生成器生成的图像尽可能不被分类为目标类别 target_class
def classification_similarity(perception_system, gen_imgs, target_class):
    gen_imgs_classified = perception_system(gen_imgs)
    predicted_classes = tf.argmax(gen_imgs_classified, axis=1)
    correct_predictions = tf.equal(predicted_classes, target_class)
    misclassification_rate = 1 - tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return misclassification_rate

# 定义指标二：相似概率
# 用frobenius_distance来度量insert_A 和 A 之间的向量距离
def similarity_probability(obj1, obj2):
    #return tf.norm(obj1 - obj2, ord='fro', axis=[-3, -2, -1]);单个维度的
    # 计算 obj1 和 obj2 的差值
    diff = obj1 - obj2
    final_norm = tf.norm(diff, ord='fro', axis=[-2, -1])

    return final_norm

# 训练 GAN
def train_gan(generator, discriminator, dataset, epochs, batch_size, perception_system, target_class):
    optimizer_G = Adam(0.0002, 0.5)
    optimizer_D = Adam(0.0002, 0.5)

    # 编译鉴别器
    discriminator.trainable = False
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 编译生成器?
    # 定义损失函数
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    #valid = discriminator(generator(input_shape))

    d_losses = []  # 存储鉴别器的损失值
    g_losses = []  # 存储生成器的损失值
    c_losses = []  # 存储分类损失值
    s_losses = []  # 存储相似概率损失值

    for epoch in range(epochs):
        for i in range(len(dataset) // batch_size):
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            real_images = dataset[idx]

            # 生成假图像
            noise = np.random.normal(0, 1, (batch_size, 100))

            # 训练判别器
            fake_images = generator.predict(noise)
            d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            with tf.GradientTape() as gen_tape:
                gen_imgs = generator(noise, training=True)
                pred_fake = discriminator(gen_imgs)
                gan_loss = cross_entropy(tf.ones_like(pred_fake), pred_fake)

                # 攻击链
                gen_sub = gen_imgs[0]
                length = gen_sub.shape[0]
                mask = generate_mask(length, 5)
                # 提取对应的子矩阵
                insert_A = insert(gen_sub, mask, orig_A)  # Using 5 insertion points
                S = insert_A.shape[0]
                timestamp1 = np.arange(0, S * 0.1, 0.1)
                A_end = resample_bfi(insert_A, timestamp1, target_freq=80)
                A_end = np.expand_dims(A_end, axis=0)

                # 添加误分类概率损失
                class_loss =classification_similarity(perception_system, A_end, target_class)
                class_loss = tf.cast(class_loss, tf.float32)  # 转换为 float32
                # 添加相似概率损失
                similarity_loss = similarity_probability(A_end, A_new)
                similarity_loss = tf.cast(similarity_loss, tf.float32)  # 转换为 float32

                # 综合损失
                c_lamda=1.5
                s_lamda=0.7
                total_gen_loss = gan_loss + c_lamda * class_loss + similarity_loss * s_lamda

            gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
            optimizer_G.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

            # 存储损失值
            d_losses.append(d_loss[0])
            # 当然，这里也可以选择g_losses.append(total_gan_loss),看想要的是什么
            g_losses.append(gan_loss.numpy())
            s_losses.append(similarity_loss)
            c_losses.append(class_loss)

            # 绘制损失图像
            epochs = range(1, len(d_losses) + 1)
            plt.figure(figsize=(12, 6))
            plt.plot(epochs, d_losses, label='Discriminator Loss')
            plt.plot(epochs, g_losses, label='Generator Loss')
            plt.title('GAN Losses During Training')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            # 打印损失
            if i % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(dataset) // batch_size} "
                      f"D_loss: {d_loss[0]} G_loss: {g_loss}")


    # 保存训练好的模型
    generator.save('./generator_model.h5')
    discriminator.save('./discriminator_model.h5')
    return d_losses, g_losses, c_losses, s_losses

# 测试 GAN
def test_gan(dataset, batch_size, perception_system, target_class, imitate_class):
    # 加载训练好的生成器和判别器
    generator = load_model('./generator_model.h5')
    discriminator = load_model('./discriminator_model.h5')

    # 加载测试数据(此处自选）
    test_data = dataset[:batch_size]  # 假设使用前 batch_size 个样本作为测试数据

    # 生成假图像
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_A = generator.predict(noise)
    # 保存生成的假图像
    # np.save('./gen.npy', fake_A)

    # 构建完整攻击链
    A_sub = fake_A[0]
    length = A_sub.shape[0]
    mask = generate_mask(length, 5)
    insert_A = insert(A_sub, mask, orig_A)
    S = insert_A.shape[0]
    timestamp1 = np.arange(0, S * 0.1, 0.1)
    A_end = resample_bfi(insert_A, timestamp1, target_freq=80)
    A_end = np.expand_dims(A_end, axis=0)

    # 评估判别器(若测试重点为攻击效果（误分类率、相似度），可完全移除判别器和生成器的评估代码。保留only用于监控模型性能。)
    d_loss_real = discriminator.evaluate(test_data, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.evaluate(fake_A, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # 评估生成器（可选）
    g_loss = discriminator.evaluate(fake_A, np.ones((batch_size, 1)), verbose=0)

    #指标一：误分类概率
    s1 = classification_similarity(perception_system, A_end, target_class)
    #指标二：相似概率
    s2 = similarity_probability(A_end,A_new)
    print("\n=== 攻击效果评估报告 ===")
    print(f"误分类概率{target_class}: {s1}")
    print(f"相似度、隐蔽性{imitate_class}: {s2}")

# 主函数
def main():
    # 加载数据集
    dataset = load_dataset('./data/push.mat')
    dataset = dataset.reshape(dataset.shape[0], 25, 4, 234)
    print(f"数据集加载完成，形状: {dataset.shape}")

    # 初始化模型
    generator = build_generator()
    discriminator = build_discriminator()

    # 假设 perception_system 是一个已经训练好的分类网络
    percept_system = load_model('./target_model.h5')
    percept_system.trainable = False

    target_class = 0  # 目标误分类类别
    imitate_class = 1  # 模仿目标类别

    # 训练 GAN
    d_losses, g_losses, c_losses, s_losses = train_gan(generator, discriminator, dataset, epochs=105, batch_size=32,
                                                       perception_system=percept_system, target_class=target_class)

    # 测试 GAN，使用已经训练好的模型
    test_gan(dataset, batch_size=32, perception_system=percept_system,
             target_class=target_class, imitate_class=imitate_class)

if __name__ == "__main__":
    main()