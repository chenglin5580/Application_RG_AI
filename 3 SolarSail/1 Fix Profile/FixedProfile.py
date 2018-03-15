

class FixedProfile():

    def choose_action(self, t):

        if t <= 66:
            return 0
        elif t <= 2 * 66:
            return 70
        elif t <= 3 * 66:
            return 60
        elif t <= 4 * 66:
            return 50
        else:
            return 45
