cd ~/Documents/probaV/researches/img2img/probaV_SR
python3 probaV_sr.py -wm rdn -en 200 --train -mp rdn
python3 probaV_sr.py -wm rdn -tr -en 200 --train -mp rdn_tre
python3 probaV_sr.py -wm carn -en 200  --train -mp carn
python3 probaV_sr.py -wm carn -tr -en 200  --train -mp carn_tre
#python3 probaV_sr.py -wm basic -en 100 --train -mp basic
