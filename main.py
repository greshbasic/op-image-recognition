# saving the trouble of waiting for hefty imports if the data folder does not exist anyway
# -------------------------------------------------------
from pathlib import Path
data = Path("Data")
if not data.exists():
    exit("Data folder does not exist")
    
import imports
# -------------------------------------------------------

