# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Generate one .rst file for each module and class in the HOOMD-blue public API.

``generate-toctree`` automatically generates the Sphinx .rst files for the HOOMD-blue
API documentation. The public API is defined by the items in the `__all__` list in
each package/module for compatibility with ruff's identification of exported APIs.
"""

import sys
import inspect
import os
from pathlib import Path

def generate_member_rst(path, name, member):
    """Generate the rst file that describes a class or data element.
    
    Does not overwrite existing files. This allows files to be automatically created
    and then customized as needed. 80+% of the files should not need customization.
    """
    print(f"member: {name} in {str(path)}")

def generate_module_rst(path, module):
    """Generate the rst file that describes a module.

    Always overwrites the file to automatically update the table of contents when
    adding new classes.
    """
    full_module_name = module.__name__
    module_name = full_module_name.split('.')[-1]
    print(f"module: `{module_name}` in `{str(path)}`")

    # Alphabetize the items
    sorted_all = module.__all__.copy()
    sorted_all.sort()

    # Split the items into modules and class members
    submodules = []
    classes = []

    for member_name in sorted_all:
        member = getattr(module, member_name)
        if inspect.ismodule(member):
            submodules.append(member_name)
            generate_module_rst(path / member_name, member)

        if inspect.isclass(member):            
            classes.append(member_name)
            generate_member_rst(path, member_name, member)

        # data members should be documented directly in the module's docstring, and
        # are ignored here.

    # Generate the {module_name}.rst
    full_module_underline = '=' * len(full_module_name)

    module_rst = f"{full_module_name}\n{full_module_underline}\n"
    module_rst += f".. automodule:: {full_module_name}\n    :members:\n\n"

    if len(submodules) > 0:
        module_rst += '.. rubric:: Modules\n\n.. toctree::\n    :maxdepth: 1\n\n'
        for submodule in submodules:
            module_rst += f'    {submodule}\n'
        module_rst += '\n'
       

    if len(classes) > 0:
        module_rst += '.. rubric:: Classes\n\n.. toctree::\n    :maxdepth: 1\n\n'
        for class_name in classes:
            module_rst += f'    {module_name}/{class_name.lower()}\n'
        module_rst += '\n'

    (path.parent() / module_name + '.rst').write_text(module_rst)

if __name__ == '__main__':
    doc_dir = Path(__file__).parent
    repository_dir = doc_dir.parent
    sys.path.insert(0, str(repository_dir))

    os.environ['SPHINX'] = '1'

    import hoomd

    print('Generating API rst files')
    generate_module_rst(doc_dir / 'hoomd', hoomd)
