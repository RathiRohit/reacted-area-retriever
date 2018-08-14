# Reacted Area Retriever
- A mini project to find the surface area for which the material has been removed (total as well as area under individual hexagon) after undergoing a chemical reaction.
- Implemented in Python with OpenCV

![ss16_mod](https://raw.githubusercontent.com/RathiRohit/reacted-area-retriever/master/output/ss16_mod.jpeg)

### How does it work ?
1. Input original image
2. Binarising original image (will be used later for background removal)
3. Detecting contours in original image
4. Area based thresholding is applied to remove unwanted contours
5. Rectangular regions are formed around hexagonal materials
6. Removing the background from original image by performing Binary AND with binarised image
7. Removing Non-reacted material (white-gray coloured material) by colour based thresholding
8. Calculating percentage area of material reacted
