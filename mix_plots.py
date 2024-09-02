import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# File paths for the PNG images
file1 = 'tfm_imgs/time_vs_cost/ETTh2_96_96/all.png'
file2 = 'tfm_imgs/time_vs_cost/ETTh2_96_192/all.png'
file3 = 'tfm_imgs/time_vs_cost/ETTh2_96_336/all.png'
file4 = 'tfm_imgs/time_vs_cost/ETTh2_96_720/all.png'
titles = ['ETTh2_96_96', 'ETTh2_96_192', 'ETTh2_96_336', 'ETTh2_96_720']

# Load the images
img1 = mpimg.imread(file1)
img2 = mpimg.imread(file2)
img3 = mpimg.imread(file3)
img4 = mpimg.imread(file4)

# Create a 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Display images in the subplots with titles
axs[0, 0].imshow(img1)
axs[0, 0].set_title(titles[0], fontsize=16)
axs[0, 0].axis('off')  # Hide axes

axs[0, 1].imshow(img2)
axs[0, 1].set_title(titles[1], fontsize=16)
axs[0, 1].axis('off')

axs[1, 0].imshow(img3)
axs[1, 0].set_title(titles[2], fontsize=16)
axs[1, 0].axis('off')

axs[1, 1].imshow(img4)
axs[1, 1].set_title(titles[3], fontsize=16)
axs[1, 1].axis('off')

plt.tight_layout()

plt.savefig("tfm_imgs/time_vs_cost/summary.png")