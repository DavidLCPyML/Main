from random import randrange, choice
alphabet_all = list('qwertyuiopasdfghjklzxcvbnm0123456789QWERTYUIOPASDFGHJKLZXCVBNM')

# This function will create a random captcha string text based on above three list.
# The input parameter is the captcha text length. Default parameter is 1.
def create_captcha_text(captcha_string_size=1):
    captcha_string = ''.join((choice(alphabet_all) for i in range(captcha_string_size)))
    return captcha_string

# Generates random color and uses it as text for the captcha
def random_color():
    levels = range(32,256,32)
    return tuple(choice(levels) for _ in range(3))

# Creates an image captcha with the generated captcha text.
def create_image_captcha(captcha_text):
    image_captcha = ImageCaptcha()
    image = image_captcha.generate_image(captcha_text)

    # Adds noise to the image.
    image_captcha.create_noise_curve(image, random_color())
    image_captcha.create_noise_dots (image, random_color())

    # Save the image to a png file.
    image_file = "./" + captcha_text + ".png"
    image_captcha.write(captcha_text, image_file)
    
    ''' # DEBUGGING
    # Display the image in a matplotlib viewer.
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    # print(image_file + " has been created.")
    '''
