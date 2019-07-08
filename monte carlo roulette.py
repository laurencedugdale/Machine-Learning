# This program creates a Monte Carlo simulation for a roulette wheel. It simulates a person repeatedly betting on either red or black. The average
# win/loss per game is then calculated (in percentage terms) and compared with the theoretical value. 



import random
wins=0
losses=0
for iteration in range(1000000):
    number=random.randint(0,36)
    if number in (2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35):
        colour='black'
    elif number==0:
        colour='green'
    else:
        colour='red'
    player_choice=random.randint(1,2)
    if player_choice==1:
        colour_choice='black'
    else:
        colour_choice='red'
    if colour==colour_choice:
        wins=wins+1
    else:
        losses=losses+1
percentage_return_per_game=round(((wins-losses)/(wins+losses))*100,3)
print('The expected percentage return per game is -2.703%') 
print('The calculated percentage return per game is ',percentage_return_per_game,'%')


