import traffic_signs_code.params
import os

plt.figure(figsize=(20,20))
img_folder=os.path.join(os.getcwd(),'..','raw_data','Train','0')
print(img_folder)
for i in range(5):
    file = random.choice(os.listdir(img_folder))
    image_path= os.path.join(img_folder, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    plt.imshow(img)
