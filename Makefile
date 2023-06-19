run_api:
	uvicorn traffic_signs_code.api.api:app --reload

run_packages_from_base:
#mkdir ~/code/Matuschkalulu && cd "$_"
#git clone git@github.com:Matuschkalulu/traffic_signs.git
#cd ~/code/Matuschkalulu/traffic_signs
#pyenv virtualenv traffic_signs
#cd ~/code/Matuschkalulu/traffic_signs
#pyenv local traffic_signs
#pip install --upgrade pip
#pip install -r https://gist.githubusercontent.com/krokrob/53ab953bbec16c96b9938fcaebf2b199/raw/9035bbf12922840905ef1fbbabc459dc565b79a3/minimal_requirements.txt
#pip install -r requirements.txt
#git checkout master
#git pull origin master
#pip install -e .
#pip list |grep traffic_signs
#python -m traffic_signs_code.ml_logic.model.py
#sudo apt update
#sudo apt install -y direnv
#code ~/.zshrc
#direnv allow .
#direnv reload .
#echo $GOOGLE_APPLICATION_CREDENTIALS

create_dirs:
	mkdir raw_data
	mkdir raw_data/Train
	mkdir raw_data/Train/unreadable
	mkdir raw_data/Train/readable
	mkdir raw_data/split_data
	mkdir raw_data/crop_images
	mkdir raw_data/output_images
	mkdir raw_data/output_video
	mkdir raw_data/models
	mkdir raw_data/street_images
	mkdir raw_data/test_video
