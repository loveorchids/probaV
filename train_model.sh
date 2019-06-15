cd ~/Documents/probaV/researches/img2img/probaV_SR
#python3 probaV_sr.py -wm rdn -en 200 --train -mp rdn
#python3 probaV_sr.py -wm rdn -tr -en 200 --train -mp rdn_tre
python3 probaV_sr.py -wm carn -en 100 --train -mp carn --filters 96
python3 probaV_sr.py -wm carn -tr -en 100 --train -mp carn_tre --filters 96
#python3 probaV_sr.py -wm basic -en 100 --train -mp basic
