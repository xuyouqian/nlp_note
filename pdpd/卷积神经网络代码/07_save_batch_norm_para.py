para = range(10)


def forward(saved,current):
    return 0.9 * saved + 0.1*current

saved = 0
for p in para:
    saved = forward(saved,p)


