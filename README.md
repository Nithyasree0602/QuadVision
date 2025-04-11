🧠 QuadVision: Intelligent Image Segmentation Using QuadTree Structure

QuadVision is an advanced interactive image segmentation tool that blends the power of QuadTree decomposition, KMeans clustering, and a Hybrid model. Designed with an intuitive Streamlit interface, it includes real-time benchmarking, hover-based analysis, and webcam segmentation.

🎯 Purpose
QuadVision aims to deliver an intelligent, adaptive, and interpretable image segmentation system using the synergy of QuadTree decomposition and KMeans clustering. This hybrid technique enables:

•	Efficient image compression and data simplification.
•	Region-based analysis using recursive subdivision.
•	Scalability for large, complex images.
•	Practical applications in medical imaging, satellite remote sensing, and computer vision research.


🧪 Methodology
Overview
QuadVision intelligently segments grayscale images using a hybrid approach combining:

•	QuadTree decomposition — to adaptively segment based on regional intensity variance.
•	KMeans clustering — to globally group similar pixel intensities.
•	A hybrid method integrates both to leverage global clustering and local adaptability.


1️⃣ Preprocessing
•	Image Input: Users upload an image (.jpg, .jpeg, .png) which is converted to grayscale to   simplify intensity-based segmentation.
•	Normalization: The image is normalized and converted into a NumPy array for computation.


2️⃣ QuadTree Segmentation
QuadTree is a hierarchical partitioning method that recursively splits the image into four quadrants based on a threshold of intensity variance.
Steps:
•	Root Node Initialization: Create a QuadNode representing the whole image.
•	Region Averaging: For each node, compute the average pixel intensity.
•	Variance/Error Check: Calculate mean squared error of the region from its average.
•	Recursive Subdivision:
•	If the error exceeds the user-defined threshold, and the depth is less than the max depth, split into 4 child nodes.
•	Leaf Nodes: Once the threshold condition is satisfied or max depth is reached, stop subdivision and store the average intensity.

🔎 Outcome: Produces a segmented image with regions of similar intensity while significantly reducing data.

3️⃣ KMeans Segmentation
KMeans Clustering groups pixels into k clusters based on intensity levels.

Steps:
•	Flatten the image into a 1D array of pixel values.
•	Apply KMeans to form k clusters.
•	Replace each pixel’s value with the centroid of its cluster.
🎨 Outcome: Efficient global segmentation ideal for distinguishing distinct regions in the image.

4️⃣ Hybrid Segmentation (KMeans + QuadTree)
To combine local adaptability with global consistency, the hybrid method applies QuadTree decomposition on the KMeans-segmented image.
Steps:
•	First, apply KMeans clustering to preprocess the image.
•	Then run QuadTree decomposition on the clustered output.

🔀 Outcome: Enhances accuracy by adapting to both global intensity groups and local variance.

5️⃣ Visualization & Analysis
•	Visual Overlays: The segmentation structure is superimposed on the image using color-coded QuadTree rectangles.
•	Compression Ratio: Calculated as Original Size / Number of QuadTree Nodes, indicating segmentation efficiency.
•	Interactive Sliders: Users can modify threshold, max depth, and number of KMeans clusters in real time.
•	Export Options: Segmented images and QuadTree structure (.json) can be downloaded or saved locally.


6️⃣ Output Files & JSON Tree Export
•	Segmented Images:
•	quadtree_segmented.png
•	kmeans_segmented.png
•	hybrid_segmented.png
•	quadtree_visualized.png
•	QuadTree Structure: JSON export of the recursive node tree including coordinates, averages, and depth for each node.


💡 Advantages of QuadVision
•	Adaptive: Dynamically adjusts to image complexity.
•	Efficient: Balances accuracy and compression.
•	Interpretable: Provides both visual and structural insights via the JSON export.
•	Extensible: Can be extended for real-time segmentation or integrated into medical/satellite diagnostic tools

💼 Use Cases

 🏥 Medical Imaging:	Segmenting tumors, tissues, and abnormalities in grayscale X-rays or MRIs.
 🛰️ Satellite Imaging:	Land cover classification, forest detection, urban zoning from grayscale maps.
 🧠 AI + CV Research:	Preprocessing step for object detection, anomaly detection, or semantic segmentation.
 🖼️ Digital Art & Design:	Texture simplification, posterization, and stylized region segmentation.
 🎮 Game Development:	Procedural terrain generation using region-based partitioning.
 📉 Data Compression:	Reducing image data size while retaining structural integrity for efficient storage.

🚀 Features

 🧩 QuadTree-based grayscale segmentation
 🎨 Dynamic KMeans clustering
 🔀 Hybrid KMeans + QuadTree segmentation
 📷 Webcam image capture and segmentation
 🖼️ Side-by-side comparison slider
 🧮 Hover-based pixel average display
 📊 Real-time performance benchmarking chart
 📁 Downloadable results + QuadTree structure in JSON

📸 Demo

![App Screenshot](assets/demo_screenshot.png)

🛠️ Installation

bash:
git clone https://github.com/your-username/QuadVision_Intelligent_Image_Segmentation.git
cd QuadVision_Intelligent_Image_Segmentation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


▶️ Run the App

bash:
streamlit run app.py


📄 License

MIT License - see [LICENSE](LICENSE) for details.

🌐 Author

Nithya Sree 
[LinkedIn](https://www.linkedin.com/in/nithya-sree-r-s-621a4b255/) 


⭐ Star this repo to support the project!
"# QuadVision" 
