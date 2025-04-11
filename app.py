import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time

# --- QuadTree Classes ---
class QuadNode:
    def __init__(self, x, y, w, h, data, depth=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.data = data
        self.childs = []
        self.average = None
        self.depth = depth

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "average": float(self.average) if self.average is not None else None,
            "depth": self.depth,
            "childs": [child.to_dict() for child in self.childs]
        }

class QuadTree:
    def __init__(self, image):
        self.root = QuadNode(0, 0, image.shape[1], image.shape[0], image)
        self.node_count = 1

    def calculate_average(self, node):
        if node.data.size > 0:
            return np.mean(node.data)
        return 0

    def calculate_error(self, node, average):
        if node.data.size > 0:
            return np.mean((node.data - average) ** 2)
        return 0

    def subdivide(self, node, threshold, max_depth):
        average = self.calculate_average(node)
        node.average = average
        error = self.calculate_error(node, average)

        if error > threshold and node.depth < max_depth:
            half_w = node.w // 2
            half_h = node.h // 2
            if half_w == 0 or half_h == 0:
                return

            child1 = QuadNode(node.x, node.y, half_w, half_h, node.data[0:half_h, 0:half_w], node.depth + 1)
            child2 = QuadNode(node.x + half_w, node.y, half_w, half_h, node.data[0:half_h, half_w:node.w], node.depth + 1)
            child3 = QuadNode(node.x, node.y + half_h, half_w, half_h, node.data[half_h:node.h, 0:half_w], node.depth + 1)
            child4 = QuadNode(node.x + half_w, node.y + half_h, half_w, half_h, node.data[half_h:node.h, half_w:node.w], node.depth + 1)

            node.childs.extend([child1, child2, child3, child4])
            self.node_count += 4

            for child in node.childs:
                self.subdivide(child, threshold, max_depth)

    def build(self, threshold, max_depth):
        self.subdivide(self.root, threshold, max_depth)

    def create_segmented_image(self, shape):
        segmented = np.zeros(shape, dtype=np.uint8)
        self._fill_segmented_image(self.root, segmented)
        return segmented

    def _fill_segmented_image(self, node, segmented_image):
        if node.average is not None:
            segmented_image[node.y:node.y + node.h, node.x:node.x + node.w] = node.average
        for child in node.childs:
            self._fill_segmented_image(child, segmented_image)

    def visualize(self, image, node, depth=0):
        color_map = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        color = color_map[depth % len(color_map)]
        cv2.rectangle(image, (node.x, node.y), (node.x + node.w, node.y + node.h), color, 1)
        for child in node.childs:
            self.visualize(image, child, depth + 1)

    def export_tree_json(self):
        return json.dumps(self.root.to_dict(), indent=2)

# --- KMeans Segmentation ---
def apply_kmeans(image, k=3):
    Z = image.reshape((-1, 1)).astype(np.float32)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(Z)
    centers = np.uint8(kmeans.cluster_centers_)
    labels = kmeans.labels_.flatten()
    segmented = centers[labels].reshape(image.shape)
    return segmented

# --- Hybrid Segmentation ---
def hybrid_kmeans_quadtree(image, k=3, threshold=50, max_depth=5):
    kmeans_segmented = apply_kmeans(image, k)
    qt = QuadTree(kmeans_segmented)
    qt.build(threshold, max_depth)
    hybrid_result = qt.create_segmented_image(image.shape)
    return hybrid_result

# --- Utilities ---
def convert_image_to_bytes(img: np.ndarray, format: str = "PNG") -> bytes:
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format=format)
    return buffer.getvalue()

def get_download_link(file_bytes: bytes, filename: str, label: str):
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">{label}</a>'

# --- Streamlit Interface ---
st.set_page_config(layout="wide")
st.title("🧠 Advanced Image Segmentation: QuadTree + KMeans")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
use_webcam = st.checkbox("📷 Use Webcam Instead")

threshold = st.slider("🎚️ QuadTree Error Threshold", 10, 100, 50, 5)
max_depth = st.slider("🧱 Max Depth", 1, 10, 5)
k_clusters = st.slider("🎨 K for KMeans", 2, 10, 3)

image = None

if use_webcam:
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
else:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

if image is not None:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Performance Measurement
    start_time = time.time()
    qt = QuadTree(grayscale)
    qt.build(threshold, max_depth)
    qt_segmented = qt.create_segmented_image(grayscale.shape)
    qt_vis = cv2.cvtColor(grayscale.copy(), cv2.COLOR_GRAY2BGR)
    qt.visualize(qt_vis, qt.root)
    end_time = time.time()
    qt_time = end_time - start_time

    start_time = time.time()
    kmeans_segmented = apply_kmeans(grayscale, k_clusters)
    end_time = time.time()
    kmeans_time = end_time - start_time

    start_time = time.time()
    hybrid_segmented = hybrid_kmeans_quadtree(grayscale, k_clusters, threshold, max_depth)
    end_time = time.time()
    hybrid_time = end_time - start_time

    # Display Comparison Slider
    st.markdown("### 🔄 Comparison Slider")
    comparison_type = st.selectbox("Select comparison", ["Original vs QuadTree", "QuadTree vs KMeans", "KMeans vs Hybrid"])
    if comparison_type == "Original vs QuadTree":
        st.image([image, qt_segmented], caption=["Original", "QuadTree"], width=300)
    elif comparison_type == "QuadTree vs KMeans":
        st.image([qt_segmented, kmeans_segmented], caption=["QuadTree", "KMeans"], width=300)
    elif comparison_type == "KMeans vs Hybrid":
        st.image([kmeans_segmented, hybrid_segmented], caption=["KMeans", "Hybrid"], width=300)

    # Hover Average (Simulated on Click)
    st.markdown("### 🖱️ Region Average")
    x = st.number_input("X coordinate", min_value=0, max_value=grayscale.shape[1] - 1)
    y = st.number_input("Y coordinate", min_value=0, max_value=grayscale.shape[0] - 1)
    avg_val = qt_segmented[int(y), int(x)]
    st.info(f"Average intensity at ({int(x)}, {int(y)}): {avg_val}")

    # Performance Plot
    st.markdown("### ⏱️ Performance Plot")
    fig, ax = plt.subplots()
    ax.bar(["QuadTree", "KMeans", "Hybrid"], [qt_time, kmeans_time, hybrid_time], color=["blue", "green", "purple"])
    ax.set_ylabel("Time (s)")
    ax.set_title("Segmentation Time Comparison")
    st.pyplot(fig)

    # Display Images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="📷 Original Image", use_column_width=True)
    with col2:
        st.image(qt_segmented, caption="🧩 QuadTree Segmented", use_column_width=True)
    with col3:
        st.image(qt_vis, caption="📐 QuadTree Visualization", use_column_width=True)

    col4, col5 = st.columns(2)
    with col4:
        st.image(kmeans_segmented, caption="🎨 KMeans Segmented", use_column_width=True)
    with col5:
        st.image(hybrid_segmented, caption="🔀 Hybrid (KMeans + QuadTree)", use_column_width=True)

    st.markdown("### 📊 Stats")
    st.markdown(f"- Total QuadTree Nodes: `{qt.node_count}`")
    st.markdown(f"- Max Depth: `{max_depth}`")
    st.markdown(f"- Compression Ratio: `{grayscale.size / qt.node_count:.2f}`")

    st.markdown("### 💾 Downloads")
    st.markdown(get_download_link(convert_image_to_bytes(qt_segmented), "quadtree_segmented.png", "📥 QuadTree"), unsafe_allow_html=True)
    st.markdown(get_download_link(convert_image_to_bytes(qt_vis), "quadtree_visualized.png", "📥 QuadTree Visualized"), unsafe_allow_html=True)
    st.markdown(get_download_link(convert_image_to_bytes(kmeans_segmented), "kmeans_segmented.png", "📥 KMeans"), unsafe_allow_html=True)
    st.markdown(get_download_link(convert_image_to_bytes(hybrid_segmented), "hybrid_segmented.png", "📥 Hybrid"), unsafe_allow_html=True)

    json_data = qt.export_tree_json()
    st.download_button("📁 Download QuadTree Structure (JSON)", data=json_data, file_name="quadtree_structure.json")

    if st.checkbox("📂 Save Images to Disk"):
        cv2.imwrite("quadtree_segmented.png", qt_segmented)
        cv2.imwrite("quadtree_visualized.png", qt_vis)
        cv2.imwrite("kmeans_segmented.png", kmeans_segmented)
        cv2.imwrite("hybrid_segmented.png", hybrid_segmented)
        st.success("✅ Images saved successfully.")
