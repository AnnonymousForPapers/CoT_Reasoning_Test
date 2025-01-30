import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Tokens list
tokens = ['Q', ':', 'ĠEvery', 'Ġimp', 'us', 'Ġis', 'Ġnot', 'Ġfloral', '.', 'ĠAlex', 'Ġis', 'Ġan', 'Ġimp', 'us', '.', 'ĠPro', 've', ':', 'ĠAlex', 'Ġis', 'Ġnot', 'Ġfloral', '.', 
          'Ċ', 'A', ':', 'ĠAlex', 'Ġis', 'Ġan', 'Ġimp', 'us', '.', 'ĠEvery', 'Ġimp', 'us', 'Ġis', 'Ġnot', 'Ġfloral', '.', 'ĠAlex', 'Ġis', 'Ġnot', 'Ġfloral', '.', 
          'Ċ', 'Ċ', 'Q', ':', 'ĠEvery', 'Ġj', 'omp', 'us', 'Ġis', 'Ġnot', 'Ġloud', '.', 'ĠRex', 'Ġis', 'Ġa', 'Ġj', 'omp', 'us', '.', 'ĠPro', 've', ':', 'ĠRex', 'Ġis', 'Ġnot', 'Ġloud', '.', 
          'Ċ', 'A', ':', 'ĠRex', 'Ġis', 'Ġa', 'Ġj', 'omp', 'us', '.', 'ĠEvery', 'Ġj', 'omp', 'us', 'Ġis', 'Ġnot', 'Ġloud', '.', 'ĠRex', 'Ġis', 'Ġnot', 'Ġloud', '.', 
          'Ċ', 'Ċ', 'Q', ':', 'ĠEach', 'Ġt', 'ump', 'us', 'Ġis', 'Ġnot', 'Ġliquid', '.', 'ĠRex', 'Ġis', 'Ġa', 'Ġt', 'ump', 'us', '.', 'ĠPro', 've', ':', 'ĠRex', 'Ġis', 'Ġnot', 'Ġliquid', '.', 
          'Ċ', 'A', ':', 'ĠRex', 'Ġis', 'Ġa', 'Ġt', 'ump', 'us', '.', 'ĠEach', 'Ġt', 'ump', 'us', 'Ġis', 'Ġnot', 'Ġliquid', '.', 'ĠRex', 'Ġis', 'Ġnot', 'Ġliquid', '.', 
          'Ċ', 'Ċ', 'Q', ':', 'ĠSter', 'p', 'uses', 'Ġare', 'Ġtransparent', '.', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠPro', 've', ':', 'ĠW', 'ren', 'Ġis', 'Ġtransparent', '.', 
          'Ċ', 'A', ':', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠSter', 'p', 'uses', 'Ġare', 'Ġtransparent', '.', 'ĠW', 'ren', 'Ġis']# , 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠPro', 've', ':', 'ĠW', 'ren', 'Ġis', 'Ġtransparent', '.', 'Ċ', 'Ċ', 'A', ':', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠSter', 'p', 'uses', 'Ġare', 'Ġtransparent', '.', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠPro', 've', ':', 'ĠW', 'ren', 'Ġis', 'Ġtransparent', '.', 'Ċ', 'Ċ', 'A', ':', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠSter', 'p', 'uses', 'Ġare', 'Ġtransparent', '.', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠPro', 've', ':', 'ĠW', 'ren', 'Ġis', 'Ġtransparent', '.', 'Ċ', 'Ċ', 'A', ':', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠSter', 'p', 'uses', 'Ġare', 'Ġtransparent', '.', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠPro', 've', ':', 'ĠW', 'ren', 'Ġis', 'Ġtransparent', '.', 'Ċ', 'Ċ', 'A', ':', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠSter', 'p', 'uses', 'Ġare', 'Ġtransparent', '.', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠPro', 've', ':', 'ĠW', 'ren', 'Ġis', 'Ġtransparent', '.', 'Ċ', 'Ċ', 'A', ':', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠSter', 'p', 'uses', 'Ġare', 'Ġtransparent', '.', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠPro', 've', ':', 'ĠW', 'ren', 'Ġis', 'Ġtransparent', '.', 'Ċ', 'Ċ', 'A', ':', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠSter', 'p', 'uses', 'Ġare', 'Ġtransparent', '.', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠPro', 've', ':', 'ĠW', 'ren', 'Ġis', 'Ġtransparent', '.', 'Ċ', 'Ċ', 'A', ':', 'ĠW', 'ren', 'Ġis', 'Ġa', 'Ġster', 'p', 'us', '.', 'ĠSter', 'p', 'uses', 'Ġare', 'Ġtransparent', '.', 'ĠW', 'ren', 'Ġis', 'Ġa']


# Simulated values for tokens (random values between 0 and 1)
values = np.random.rand(len(tokens))

# Define image properties
img_width = 1000  # Width of the image
padding = 1
line_height = 25
font_size = 25

# Calculate number of lines needed based on occurrences of 'Ċ'
num_lines = tokens.count('Ċ') + 2
img_height = num_lines * line_height + padding * 2

# Create an image canvas
image = Image.new("RGB", (img_width + 80, img_height), "white")  # Adjust space for smaller color bar
draw = ImageDraw.Draw(image)

# Load a font (use default if not found)
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
    font = ImageFont.load_default()

# Position tracking
x_pos, y_pos = padding, padding

for token, value in zip(tokens, values):
    # Handle new line on encountering Ċ
    if token == 'Ċ':
        x_pos = padding  # Reset X position
        y_pos += line_height  # Move to the next line
        continue

    # Replace Ġ with a space
    token = token.replace('Ġ', ' ')

    # Convert value (0 to 1) to a shade of blue (RGB format)
    color = (int(255 * (1 - value)), int(255 * (1 - value)), 255)

    # Calculate text size
    text_width, text_height = draw.textsize(token, font=font)
    text_height = 24
    
    # Move to the next line only if there's a Ċ
    if x_pos + text_width + padding > img_width - 0:
        x_pos = padding  # Reset X position
        y_pos += line_height  # Move to the next line

    # Draw background rectangle with blue shades
    draw.rectangle([x_pos+4, y_pos, x_pos + text_width + padding+3, y_pos + text_height], fill=color)
    
    # Decide text color based on background brightness
    brightness = np.mean(color)
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

    # Draw text on top of colored background
    draw.text((x_pos + 5, y_pos), token, fill=text_color, font=font)
    
    # Update X position for the next token
    x_pos += text_width + padding

# Set the desired height for the color bar (e.g., 70% of image height)
color_bar_height = int(img_height * 0.7)  # 70% of the image height
color_bar_top = int((img_height - color_bar_height) / 2)  # Position it at the middle

# Create a smaller color bar to visualize blue intensity scale (0 at bottom to 1 at top)
bar_width = 20
color_bar_x = img_width + 20  # Adjust position of the bar
for i in range(101):
    # Calculate shade based on the blue intensity
    shade = int(255 * (i / 100))
    bar_color = (shade, shade, 255)
    
    # Draw each rectangle for the color bar
    draw.rectangle([color_bar_x, color_bar_top + i * (color_bar_height / 100), 
                    color_bar_x + bar_width, color_bar_top + (i + 1) * (color_bar_height / 100)], 
                   fill=bar_color)

# Draw a black border around the color bar
draw.rectangle([color_bar_x - 2, color_bar_top, color_bar_x + bar_width + 2, color_bar_top + color_bar_height], 
               outline="black", width=3)

# Add color bar labels (0 at bottom and 1 at top)
draw.text((color_bar_x + bar_width + 10, color_bar_top + color_bar_height - 10), "0", fill="black", font=font)
draw.text((color_bar_x + bar_width + 10, color_bar_top - 10), "1", fill="black", font=font)

# Save and display the image
image.show()
# image.save("tokens_blue_gradient_with_black_border.png")

