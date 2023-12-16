sudo apt install python3.10-venv

pip3 install pyautogui
pip3 install imageio
pip3 install imageio[ffmpeg]
pip3 install selenium

sudo apt-get install python3-tk python3-dev

sudo apt install gnome-screenshot

## Install Chrome on Linux

##### Download and install the Google Chrome signing key
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -

##### Add the Google Chrome repository to the system
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'

##### Update the package list
sudo apt update

# sudo apt install -y google-chrome-stable

# Install wget (if not already installed)
sudo apt install -y wget

# Download and extract ChromeDriver
# wget https://chromedriver.storage.googleapis.com/$(wget -qO- https://chromedriver.storage.googleapis.com/LATEST_RELEASE)/chromedriver_linux64.zip
# unzip chromedriver_linux64.zip

