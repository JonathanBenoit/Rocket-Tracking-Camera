# A la place d'utiliser plein de if-else, j'ai trouvé que l'on peut faire un dictionnaire (ça ressemblerait à ça)
def get_motor_instruction(position):
    instructions = {
        'Centered': 'Up',
        'bottom': 'Accelerate Up',
        'top': 'Decelerate Up',
        'bottom left': 'Accelerate Up and Right',
        'bottom right': 'Accelerate Up and Left',
        'left': 'Up and Right',
        'right': 'Up and Left',
        'top left': 'Decelerate Up and Right',
        'top right': 'Decelerate Up and Left'
    }
    return instructions.get(position, 'Invalid position')