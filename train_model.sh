cd ~/Documents/probaV/researches/img2img/probaV_SR
python3 probaV_sr.py -wm rdn -en 100 --train
python3 probaV_sr.py -wm rdn -tr -en 100 --train
python3 probaV_sr.py -wm carn -en 100  --train
python3 probaV_sr.py -wm carn -tr -en 100  --train
python3 probaV_sr.py -wm basic -en 100 --train
