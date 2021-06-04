
def plot_data(data, ax, width=0.5, colors=['blue','red']):
    for i, entry in enumerate(data):
        h1 = entry['Latent OOD error']
        h2 = h1 + entry['Test Information Loss']
        ax.fill_between([i-width/2.0, i+width/2.0],
                         [0,0],
                         [h1]*2, label='Latent OOD error' if i==0 else None,
                       color=colors[0])
        ax.fill_between([i-width/2.0, i+width/2.0],
                         [h1]*2, 
                         [h2]*2, label='Test Information loss' if i==0 else None,
                       color=colors[1])
        
    ax.set_ylim(0)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([entry['Model'] for entry in data], rotation=90)
    ax.set_xlim(-1,10)
    ax.set_ylabel('OOD error', size=15)
    ax.legend()
     