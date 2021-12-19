import numpy as np 
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import plot_confusion_matrix
from tqdm import tqdm 
import code.preparation as prep
import code.visualization as viz
np.random.seed(123)

header = st.container()
dataset = st.container()
features = st.container()
modeltraining = st.container()

with header:
    st.title('Welcome to the World Development Mental Health Acquisition!')
    st.markdown(
    """
    <style>
   .sidebar .sidebar-content {
        background: #FFED91;
    }
    </style>
    """,
    unsafe_allow_html=True
)