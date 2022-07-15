from ipywidgets import widgets, Layout, Button


def get_text_input_smiles():
    smiles_input = widgets.Textarea(
        value='CCC1(c2ccccc2)C(=O)NC(=O)NC1=O\nCC(O)C(=O)O\nO=C(O)c1c(Cl)cccc1Cl',
        placeholder='CCC1(c2ccccc2)C(=O)NC(=O)NC1=O',
        description='',
        rows=5,
        layout=widgets.Layout(height="4", width="auto"),
    )

    indicator_text = widgets.Label(value='', layout=widgets.Layout(height="4", width="auto"),
                                   style={"font-weight": "bold"})

    submit_button = widgets.Button(description='Submit', button_style='success',
                                   layout=Layout(width='30%', height='80px'))

    def on_button_clicked(b):
        submit_button.button_style = "warning"
        submit_button.description = "Reload"
        num_strings = len(smiles_input.value.split("\n"))
        indicator_text.value = f"Loaded {num_strings} SMILES strings."

    submit_button.on_click(on_button_clicked)

    return smiles_input, submit_button, indicator_text



def get_buttons_upload_file():
    uploader = widgets.FileUpload(
        accept='',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
        multiple=False,  # True to accept multiple files upload else False,
        layout=Layout(width='30%', height='80px'),
        description="Upload CSV or SDF"
    )

    upload_button = Button(
        description='Use current file',
                    button_style='success',
                    layout=Layout(width='10%', height='80px'),
                    icon="fa-floppy-o",
                    tooltip='Click me'
    )

    reset_button = Button(
        description='Reset',
        button_style='danger',
        layout=Layout(width='10%', height='80px'),
        icon="fa-floppy-o",
        tooltip='Click me'
    )

    output = widgets.Output()

    @output.capture()
    def on_button_use_loaded_clicked(b):
        if len(uploader.value):
            b.description = "File fixed"
            b.icon = "check"
            b.button_style = 'warning'
            uploader.description = str(*list(uploader.value.keys()))

    @output.capture()
    def on_button_reset_clicked(b):
        uploader.value.clear()
        uploader._counter = 0
        uploader.description = "Upload CSV or SDF"

        upload_button.description = 'Use current file'
        upload_button.button_style = 'success'
        upload_button.icon = "fa-floppy-o"
        upload_button.value = ""

    upload_button.on_click(on_button_use_loaded_clicked)
    reset_button.on_click(on_button_reset_clicked)

    return uploader, upload_button, reset_button


def get_text_column_name():
    column_name = widgets.Text(
        value='SMILES',
        placeholder='SMILES',
        description='SMILES Column Name',
        disabled=False,
        layout=Layout(width='30%', height='80px'),
        style={'description_width': 'initial'}
    )
    return column_name



