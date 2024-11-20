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

def generate_member_rst(path, full_module_name, name, type):
    """Generate the rst file that describes a class or data element.
    
    Does not overwrite existing files. This allows files to be automatically created
    and then customized as needed. 80+% of the files should not need customization.
    """
    print(f"member: {name} in {str(path)}")

    # Generate the file {name}.rst
    underline = '=' * len(name)

    member_rst = f"{name}\n{underline}\n\n"
    member_rst += f".. py:currentmodule:: {full_module_name}\n\n"
    
    member_rst += f".. auto{type}:: {name}\n"

    destination = (path / name.lower()).with_suffix('.rst')

    # TODO: uncomment when testing is complete    
    # if destination.exists():
    #     return

    destination.write_text(member_rst)
        

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
   
    if len(sorted_all) > 0:
        path.mkdir(exist_ok=True)

    # Split the items into modules and class members
    submodules = []
    classes = []
    functions = []

    for member_name in sorted_all:
        member = getattr(module, member_name)
        if inspect.ismodule(member):
            submodules.append(member_name)
            generate_module_rst(path / member_name, member)

        if inspect.isclass(member):            
            classes.append(member_name)
            generate_member_rst(path, full_module_name, member_name, 'class')

        if inspect.isfunction(member):            
            functions.append(member_name)
            generate_member_rst(path, full_module_name, member_name, 'function')

        # data members should be documented directly in the module's docstring, and
        # are ignored here.

    # Generate the file module-{module_name}.rst
    module_underline = '=' * len(module_name)

    module_rst = f"{module_name}\n{module_underline}\n\n"
    module_rst += f".. automodule:: {full_module_name}\n"
    module_rst += "    :members:\n"
    module_rst += f"    :exclude-members: {','.join(classes + functions)}\n\n"

    if len(submodules) > 0:
        module_rst += '.. rubric:: Modules\n\n.. toctree::\n    :maxdepth: 1\n\n'
        for submodule in submodules:
            module_rst += f'    {module_name}/module-{submodule}\n'
        module_rst += '\n'
       

    if len(classes) > 0:
        module_rst += '.. rubric:: Classes\n\n.. toctree::\n    :maxdepth: 1\n\n'
        for class_name in classes:
            module_rst += f'    {module_name}/{class_name.lower()}\n'
        module_rst += '\n'

    if len(functions) > 0:
        module_rst += '.. rubric:: Functions\n\n.. toctree::\n    :maxdepth: 1\n\n'
        for function_name in functions:
            module_rst += f'    {module_name}/{function_name.lower()}\n'
        module_rst += '\n'

    (path.parent / ('module-' + module_name)).with_suffix('.rst').write_text(module_rst)

if __name__ == '__main__':
    doc_dir = Path(__file__).parent
    repository_dir = doc_dir.parent
    sys.path.insert(0, str(repository_dir))

    os.environ['SPHINX'] = '1'

    import hoomd

    print('Generating API rst files')
    generate_module_rst(doc_dir / 'hoomd', hoomd)
