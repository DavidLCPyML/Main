NUM_OF_LETTERS = 4
NUM_SAMPLES = 50000
# LEFT_BOUND = 6
# RIGHT_BOUND = 10

# generate a dataset of captchas for our model to train with.
for i in range(NUM_SAMPLES):
    # captcha_text = create_random_captcha_text(randrange(LEFT_BOUND,RIGHT_BOUND))
    captcha_text = create_captcha_text(NUM_OF_LETTERS)
    create_image_captcha(captcha_text)
    # print(captcha_text)    
    
    '''# DEBUGGING PURPOSES
    img = create_image_captcha(captcha_text)
    print(captcha_text)
    '''
