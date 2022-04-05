#THIS CODE WAS USED TO HELP ENHANCE THE DATASET. 
#IMAGES ARE RANGED 100 AT A TIME BECAUSE OTHERWISE THE PROCESS IS KILLED BY THE OS
#IT CREATES 4 VERSIONS,ONE 500*500 , ONE ORIGINAL SIZE BUT BRIGHTENED, ONE ORIGINAL SIZE BUT DARKENED,ONE ORIGINAL SIZE BUT ROTATED 90 DEGREES
#REPLACE HELL WITH YOUR FOLDER NAME


from PIL import Image,ImageEnhance
import os
import PIL
import glob
import numpy as np
images = []
for f in glob.iglob("images/hell/*"):
    images.append((Image.open(f)))

# images = np.array(images)

#resize
for i in range(0,100):
    resized_image = images[i].resize((500,500))
    resized_image.save('images/hell500/'+str(i)+'.jpg')

    # brighten
    enhancer = ImageEnhance.Brightness(images[i])
    enhanced_image = enhancer.enhance(1.5)
    resized_image.save('images/hell500/bright'+str(i)+'.jpg')
    #darken
    enhanced_image = enhancer.enhance(0.5)
    resized_image.save('images/hell500/dark'+str(i)+'.jpg')
    
    rotated_image=images[i].rotate(90)
    rotated_image.save('images/hell500/rotated'+str(i)+'.jpg')


for i in range(100,200):
    resized_image = images[i].resize((500,500))
    resized_image.save('images/hell500/'+str(i)+'.jpg')

    # brighten
    enhancer = ImageEnhance.Brightness(images[i])
    enhanced_image = enhancer.enhance(1.5)
    resized_image.save('images/hell500/bright'+str(i)+'.jpg')
    #darken
    enhanced_image = enhancer.enhance(0.5)
    resized_image.save('images/hell500/dark'+str(i)+'.jpg')
    
    rotated_image=images[i].rotate(90)
    rotated_image.save('images/hell500/rotated'+str(i)+'.jpg')

for i in range(200,300):
    resized_image = images[i].resize((500,500))
    resized_image.save('images/hell500/'+str(i)+'.jpg')

    # brighten
    enhancer = ImageEnhance.Brightness(images[i])
    enhanced_image = enhancer.enhance(1.5)
    resized_image.save('images/hell500/bright'+str(i)+'.jpg')
    #darken
    enhanced_image = enhancer.enhance(0.5)
    resized_image.save('images/hell500/dark'+str(i)+'.jpg')
    
    rotated_image=images[i].rotate(90)
    rotated_image.save('images/hell500/rotated'+str(i)+'.jpg')


for i in range(300,400):
    resized_image = images[i].resize((500,500))
    resized_image.save('images/hell500/'+str(i)+'.jpg')

    # brighten
    enhancer = ImageEnhance.Brightness(images[i])
    enhanced_image = enhancer.enhance(1.5)
    resized_image.save('images/hell500/bright'+str(i)+'.jpg')
    #darken
    enhanced_image = enhancer.enhance(0.5)
    resized_image.save('images/hell500/dark'+str(i)+'.jpg')
    
    rotated_image=images[i].rotate(90)
    rotated_image.save('images/hell500/rotated'+str(i)+'.jpg')




