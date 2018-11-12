import json
import matplotlib.pyplot as plt


plots = [
    {
        'path': 'U02.1',
        'file': 'learning_curves.json',
        'curves' : [
            {'id': (0, 0, 0), 'label': 'vanilla'}
        ]
    },

    {
        'path': 'U02.2',
        'file': 'learning_curves_0.json',
        'curves' : [
            {'id': (0, 0, 0), 'label': 'lr=0.1'},
            {'id': (1, 0, 0), 'label': 'lr=0.01'},
            {'id': (2, 0, 0), 'label': 'lr=0.001'},
            {'id': (3, 0, 0), 'label': 'lr=0.0001'},
        ]
    },

    {
        'path': 'U02.3',
        'file': 'learning_curves_0.json',
        'curves' : [
            {'id': (0, 0, 0), 'label': 'filter size=1'},
            {'id': (1, 0, 0), 'label': 'filter size=3'},
            {'id': (2, 0, 0), 'label': 'filter size=5'},
            {'id': (3, 0, 0), 'label': 'filter size=7'},
        ]
    },

    {
        'path': 'U02.4',
        'file': 'learning_curves_1.json',
        'curves' : [
            {'id': (0, 0, 0), 'label': 'incumbent'},
        ]
    }
]


for plot in plots:
    json_data=open('./results/' + plot['path'] + '/' + plot['file']).read()

    data = json.loads(json_data)

    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    tlos_fig = fig.add_subplot(2, 2, 1)
    tlos_fig.set_title('Train Loss')
    tlos_fig.set_ylabel('loss')
    tlos_fig.set_xlabel('epochs')
    tlos_fig.grid(True)

    tacc_fig = fig.add_subplot(2, 2, 2)
    tacc_fig.set_title('Train Accuracy')
    tacc_fig.set_ylabel('accuracy')
    tacc_fig.set_xlabel('epochs')
    tacc_fig.grid(True)

    vlos_fig = fig.add_subplot(2, 2, 3)
    vlos_fig.set_title('Validation Loss')
    vlos_fig.set_ylabel('loss')
    vlos_fig.set_xlabel('epochs')
    vlos_fig.grid(True)

    vacc_fig = fig.add_subplot(2, 2, 4)
    vacc_fig.set_title('Validation Accuracy')
    vacc_fig.set_ylabel('accuracy')
    vacc_fig.set_xlabel('epochs')
    vacc_fig.grid(True)

    for curve in plot['curves']:
        id = data[str(curve['id'])]

        tlos_fig.plot(id['loss'], label=curve['label'])
        vlos_fig.plot(id['val_loss'], label=curve['label'])

        tacc_fig.plot(id['acc'], label=curve['label'])
        vacc_fig.plot(id['val_acc'], label=curve['label'])


    tlos_fig.legend(bbox_to_anchor=(0., -.15, 1., .102), loc='lower left', fontsize='xx-small',ncol=2)
    vlos_fig.legend(bbox_to_anchor=(0., -.15, 1., .102), loc='lower left', fontsize='xx-small', ncol=2)
    tacc_fig.legend(bbox_to_anchor=(0., -.15, 1., .102), loc='lower left', fontsize='xx-small', ncol=2)
    vacc_fig.legend(bbox_to_anchor=(0., -.15, 1., .102), loc='lower left', fontsize='xx-small', ncol=2)


    fig.savefig('./results/' + plot['path'] + "/report_learning_curves.png",dpi=300)
