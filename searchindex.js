Search.setIndex({"docnames": ["basics/builders", "basics/model_hierarchy", "basics/naming_conventions", "basics/tagged_promotion", "developers/mphys_group", "developers/new_multiphysics_problems", "index", "references/papers_using_mphys", "scenarios/aerodynamic", "scenarios/aerostructural", "scenarios/structural"], "filenames": ["basics/builders.rst", "basics/model_hierarchy.rst", "basics/naming_conventions.rst", "basics/tagged_promotion.rst", "developers/mphys_group.rst", "developers/new_multiphysics_problems.rst", "index.rst", "references/papers_using_mphys.rst", "scenarios/aerodynamic.rst", "scenarios/aerostructural.rst", "scenarios/structural.rst"], "titles": ["Builders", "Model Hierarchy", "Variable Naming Conventions", "Tagged Promotion", "The MphysGroup", "Extending the Scenario Library", "Documentation for MPhys", "Papers Using MPhys", "Aerodynamic Scenario", "Aerostructural Scenario", "Structural Scenario"], "terms": {"In": [0, 1, 4, 5, 7], "larg": [0, 6], "multiphys": [0, 1, 2, 5], "problem": [0, 1, 2, 5, 6, 9], "creation": 0, "connect": [0, 1, 3], "openmdao": [0, 1, 2, 4, 6], "can": [0, 1, 3, 4, 5, 9], "complic": 0, "time": [0, 5], "consum": 0, "The": [0, 1, 3, 5, 6, 8, 9, 10], "design": [0, 1, 3, 7], "mphy": [0, 1, 2, 3, 4, 5, 8, 9, 10], "i": [0, 1, 2, 4, 5, 6, 8, 9, 10], "base": [0, 4, 7, 8, 9, 10], "class": [0, 1, 4, 5], "order": [0, 6, 8, 9, 10], "reduc": 0, "burden": 0, "user": [0, 1, 3, 5], "most": [0, 5], "assembli": 0, "model": [0, 6, 7], "handl": [0, 7], "set": [0, 1, 2, 4, 5, 6, 8, 9, 10], "helper": [0, 1], "object": [0, 1, 3], "develop": [0, 5], "wish": [0, 3], "integr": 0, "code": [0, 2, 6], "should": [0, 1, 4, 5], "subclass": [0, 1, 4, 5], "implement": [0, 4, 5, 9], "method": [0, 1, 3, 4, 5, 7, 9], "relev": 0, "Not": 0, "all": [0, 5], "need": [0, 1, 3, 4, 5, 8, 9, 10], "For": [0, 1, 9], "exampl": [0, 1, 9, 10], "transfer": [0, 1, 6, 7], "scheme": [0, 1], "mai": 0, "precoupl": 0, "post": [0, 5, 8, 9, 10], "coupl": [0, 2, 3, 4, 6, 7, 8, 9, 10], "subsystem": [0, 1, 3, 4, 5, 8, 9, 10], "scenario": [0, 3, 4], "sourc": [0, 1, 3, 4, 5], "templat": 0, "creat": [0, 1], "becaus": [0, 1, 4, 9], "mpi": [0, 1], "commun": [0, 1], "us": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10], "insid": [0, 8, 9, 10], "known": 0, "when": [0, 2], "instanti": [0, 1, 5], "actual": 0, "solver": [0, 1, 2, 4, 5, 6, 7], "etc": 0, "constructor": 0, "initi": [0, 1], "comm": [0, 1], "thi": [0, 1, 2, 3, 5, 6, 8, 9, 10], "call": [0, 1, 4, 5], "avail": [0, 6], "paramet": [0, 1, 5], "xfer": 0, "instanc": [0, 4], "get_mesh_coordinate_subsystem": 0, "scenario_nam": [0, 3], "none": [0, 1, 5, 9], "contain": [0, 1], "return": [0, 9], "mesh": [0, 1, 3, 9], "coordin": [0, 1, 2, 3, 9], "str": [0, 1, 5, 8, 9, 10], "name": [0, 1, 4, 5, 6], "compon": [0, 1, 3, 5, 7, 8, 9, 10], "group": [0, 4, 6, 8, 9, 10], "ha": [0, 1, 6], "an": [0, 1, 3, 4, 5, 7, 8, 9, 10], "output": [0, 3, 9], "get_coupling_group_subsystem": [0, 5, 9], "add": [0, 1, 4, 5, 6, 9], "couplinggroup": [0, 1, 3, 4, 5], "comput": [0, 1, 3, 7, 9], "multipl": 0, "get_pre_coupling_subsystem": 0, "ad": [0, 1, 4, 5, 8, 9, 10], "each": [0, 1, 5, 6, 9], "befor": [0, 1], "get_post_coupling_subsystem": 0, "after": [0, 1], "get_number_of_nod": 0, "number": [0, 1, 2], "node": [0, 9], "defin": [0, 5], "interfac": [0, 2, 6], "input": [0, 1, 3, 5], "number_of_nod": 0, "int": 0, "domain": [0, 7], "get_ndof": [0, 9], "degre": [0, 9], "freedom": [0, 9], "locat": 0, "ndof": 0, "get_tagged_indic": 0, "tag": [0, 1, 2, 4, 5, 6], "grid": 0, "id": 0, "list": [0, 1], "bodi": 0, "boundari": [0, 7], "integ": 0, "string": [0, 8, 9, 10], "grid_id": 0, "correspond": 0, "given": [0, 1, 6, 9], "pattern": [1, 5], "build": 1, "optim": [1, 7], "level": [1, 3], "differ": [1, 2, 3], "type": [1, 3, 5, 6, 8, 9, 10], "provid": [1, 2, 6], "highest": 1, "consist": 1, "which": [1, 8, 9, 10], "repres": [1, 3, 5], "condit": [1, 5], "analys": 1, "perform": [1, 5], "within": 1, "primari": [1, 9], "builder": [1, 5, 6, 8, 10], "ar": [1, 3, 4, 6, 8, 9, 10], "help": [1, 2, 5], "popul": 1, "from": [1, 3, 4, 5, 9, 10], "promot": [1, 4, 5, 6], "specif": [1, 5], "variabl": [1, 3, 5, 6, 7], "physic": [1, 2, 6, 9], "being": [1, 6], "solv": [1, 2, 6], "That": 1, "modul": [1, 6, 9], "aerodynam": [1, 2, 6, 7, 9], "structur": [1, 2, 6, 7], "potenti": [1, 7], "interpol": 1, "between": 1, "load": [1, 10], "displac": [1, 2], "typic": [1, 9], "associ": [1, 2, 5], "automat": [1, 4, 8, 9, 10], "proper": 1, "have": [1, 4, 5], "default": [1, 4, 5], "nonlinear": [1, 4, 5, 7], "linear": [1, 4, 5, 7, 8, 9, 10], "overwritten": 1, "option": [1, 5, 6], "argument": [1, 5], "mphys_add_scenario": 1, "could": 1, "cruis": 1, "flight": 1, "requir": [1, 2, 5, 6, 8, 10], "determin": [1, 8, 9, 10], "lift": [1, 7], "drag": 1, "ani": [1, 3, 4, 5], "occur": 1, "sonic": 1, "boom": 1, "propag": 1, "flow": [1, 2], "solut": 1, "one": [1, 5], "wai": 1, "doe": [1, 6], "therefor": 1, "put": 1, "converg": 1, "librari": 1, "see": 1, "detail": [1, 6], "about": [1, 9], "standard": [1, 4, 6], "If": [1, 4, 5, 8, 9, 10], "particular": [1, 2, 3, 5], "cover": 1, "new": [1, 4, 6], "mphysgroup": [1, 6], "There": [1, 3], "two": 1, "version": 1, "deriv": 1, "parallelgroup": 1, "both": [1, 3, 9], "function": 1, "lower": 1, "sequenti": 1, "evalu": 1, "top": 1, "setup": [1, 4], "follow": [1, 6], "step": 1, "must": [1, 5, 9], "done": [1, 4], "": [1, 4, 5, 7, 8, 9, 10], "self": [1, 5], "other": [1, 3, 5], "like": [1, 3, 5], "geometri": [1, 8, 9, 10], "addition": 1, "hold": 1, "These": [1, 3, 6], "extra": 1, "kwarg": [1, 4, 5], "extens": [1, 6], "block": [1, 4, 5], "gauss": [1, 4, 5], "seidel": [1, 4, 5], "coupling_nonlinear_solv": 1, "coupling_linear_solv": 1, "nonlinearsolv": 1, "assign": 1, "primal": 1, "linearsolv": 1, "sensit": 1, "mphys_connect_scenario_coordinate_sourc": 1, "disciplin": [1, 3], "A": [1, 4, 5, 7, 8, 9, 10], "aid": 1, "target": 1, "assum": 1, "x_": 1, "0": 1, "api": 1, "rank": 1, "greater": 1, "than": [1, 4], "equal": 1, "simultan": 1, "unlik": 1, "so": 1, "in_multipointparallel": [1, 5], "true": [1, 8, 9, 10], "outsid": [1, 3], "cannot": 1, "directli": [1, 4, 5], "higher": 1, "parallel": [1, 6], "mpi_proc_alloc": 1, "while": [2, 4, 6], "possibl": 2, "up": [2, 6], "same": 2, "prefer": 2, "more": [2, 6], "easili": 2, "interchang": 2, "tabl": 2, "descript": [2, 6, 8, 9, 10], "x_aero0": 2, "mphys_coordin": [2, 3], "surfac": [2, 7, 9], "jig": [2, 7, 9], "shape": [2, 7, 9], "x_aero": 2, "mphys_coupl": [2, 3], "deform": 2, "u_aero": 2, "f_aero": 2, "forc": [2, 9], "t_convect": 2, "temperatur": 2, "convect": 2, "q_convect": 2, "heat": [2, 7], "x_struct0": 2, "u_struct": 2, "state": [2, 3, 9], "vector": [2, 9], "f_struct": 2, "t_conduct": 2, "thermal": 2, "side": 2, "q_conduct": 2, "conduct": 2, "To": [2, 5], "make": [2, 5], "swap": 2, "easier": 2, "also": [2, 5], "share": [2, 3], "noncoupl": 2, "aoa": 2, "mphys_input": [2, 3], "angl": 2, "attack": 2, "pleas": 2, "includ": [2, 5, 7, 9], "unit": 2, "deg": 2, "rad": 2, "declar": [2, 5], "yaw": 2, "mach": 2, "refer": 2, "reynold": 2, "q_inf": 2, "dynam": [2, 7], "pressur": 2, "select": 3, "four": 3, "data": 3, "among": 3, "come": 3, "manual": 3, "input_nam": 3, "onli": [3, 8, 10], "mphys_result": 3, "result": 3, "you": [3, 4, 5], "result_nam": 3, "access": 3, "constraint": [3, 7, 10], "thei": [3, 6], "further": 3, "behind": 3, "curtain": 3, "configur": [3, 4, 5, 7], "autom": [3, 4], "purpos": 4, "mechanan": 4, "mphys_add_subsystem": [4, 5], "still": 4, "add_subsystem": 4, "dure": 4, "phase": [4, 5], "your": [4, 5], "sure": 4, "parent": [4, 5], "super": [4, 5], "import": 4, "understand": [4, 7], "interact": [4, 9], "inherit": 4, "rather": 4, "mphys_group": 4, "some": 4, "appropri": 4, "get": 5, "them": 5, "balanc": 5, "suitabl": 5, "case": [5, 10], "necessari": [5, 6], "do": 5, "coupling_group": 5, "custom": 5, "least": 5, "As": 5, "mode": 5, "oper": 5, "otherwis": 5, "free": 5, "basic": 5, "mphys_add_pre_coupling_subsystem": 5, "mphys_add_post_coupling_subsystem": 5, "analysi": [5, 6, 7], "point": 5, "multipoint": [5, 7], "run": 5, "main": 5, "_mphys_scenario_setup": 5, "mphys_add_post_subsystem": 5, "promotes_input": 5, "promotes_output": 5, "end": 5, "system": [5, 7], "iter": 5, "tupl": [5, 9], "mphys_": 5, "els": 5, "packag": [6, 7], "high": [6, 7], "fidel": [6, 7], "eas": 6, "straightforward": 6, "multidisciplinari": [6, 7], "address": 6, "its": 6, "convent": 6, "absolut": 6, "guidelin": 6, "veri": 6, "gener": [6, 10], "capabl": 6, "howev": 6, "usag": 6, "modular": 6, "across": 6, "technologi": 6, "collabor": 6, "area": 6, "research": [6, 7], "strive": 6, "how": 6, "work": [6, 9], "hierarchi": 6, "describ": 6, "aerostructur": [6, 7], "page": 6, "extend": 6, "paper": 6, "index": 6, "search": 6, "1": [7, 9], "m": 7, "saja": 7, "abdul": 7, "kaiyoom": 7, "anil": 7, "yildirim": 7, "joaquim": 7, "r": 7, "martin": 7, "aeropropuls": 7, "over": 7, "wing": 7, "nacel": 7, "aiaa": 7, "scitech": 7, "forum": 7, "januari": 7, "2023": 7, "doi": 7, "10": 7, "2514": 7, "6": 7, "0327": 7, "2": [7, 9], "ran": 7, "aviat": 7, "san": 7, "diego": 7, "ca": 7, "june": 7, "3588": 7, "3": [7, 9], "josh": 7, "l": 7, "anib": 7, "charl": 7, "mader": 7, "aerotherm": 7, "x": 7, "57": 7, "motor": 7, "orlando": 7, "fl": 7, "2020": 7, "2115": 7, "4": [7, 9], "cfd": 7, "plate": 7, "fin": 7, "exchang": 7, "2022": 7, "3930": 7, "5": 7, "joshua": 7, "phd": 7, "thesi": 7, "univers": 7, "michigan": 7, "ann": 7, "arbor": 7, "http": 7, "deepblu": 7, "lib": 7, "umich": 7, "edu": 7, "2027": 7, "42": 7, "171375": 7, "electr": 7, "aircraft": 7, "conjug": 7, "intern": 7, "journal": 7, "mass": 7, "189": 7, "122689": 7, "1016": 7, "j": 7, "ijheatmasstransf": 7, "7": 7, "garo": 7, "bedonian": 7, "jason": 7, "e": 7, "hicken": 7, "adapt": 7, "sampl": 7, "gradient": 7, "enhanc": 7, "surrog": 7, "american": 7, "institut": 7, "aeronaut": 7, "astronaut": 7, "3998": 7, "8": 7, "adrien": 7, "crovato": 7, "romain": 7, "boman": 7, "vincent": 7, "terrapon": 7, "grigorio": 7, "dimitriadi": 7, "alex": 7, "p": 7, "prado": 7, "pedro": 7, "h": 7, "cabral": 7, "fast": 7, "full": [7, 9], "calcul": 7, "preliminari": 7, "aeroelast": 7, "9": 7, "alasdair": 7, "c": 7, "grai": 7, "graem": 7, "kennedi": 7, "geometr": 7, "3316": 7, "hannah": 7, "hajdik": 7, "bernardo": 7, "pacini": 7, "benjamin": 7, "brelj": 7, "combin": 7, "3589": 7, "11": 7, "ping": 7, "he": 7, "heyecan": 7, "koyuncuoglu": 7, "helen": 7, "hu": 7, "anvesh": 7, "dhulipalla": 7, "haiyang": 7, "hui": 7, "uav": 7, "propel": 7, "adjoint": 7, "0531": 7, "12": 7, "kevin": 7, "jacobson": 7, "bret": 7, "stanford": 7, "flutter": [7, 9], "constrain": 7, "frequenc": 7, "approach": 7, "2242": 7, "13": 7, "multi": 7, "1844": 7, "14": 7, "andrew": 7, "lamkin": 7, "nathan": 7, "wuki": 7, "advanc": 7, "bypass": 7, "turbofan": 7, "engin": 7, "3591": 7, "15": 7, "33rd": 7, "congress": 7, "council": 7, "scienc": 7, "septemb": 7, "16": 7, "christoph": 7, "lupp": 7, "inclus": 7, "effect": 7, "163259": 7, "17": 7, "malhar": 7, "prajapati": 7, "karthik": 7, "duraisami": 7, "urban": 7, "air": 7, "mobil": 7, "vehicl": 7, "0326": 7, "18": 7, "toward": 7, "mix": 7, "aero": [7, 9], "acoust": 7, "3905": 7, "19": 7, "distribut": 7, "propuls": 7, "nasa": 7, "tiltw": 7, "concept": 7, "0143": 7, "20": 7, "pawel": 7, "chwalowski": 7, "ongo": 7, "predict": 7, "valid": 7, "activ": 7, "langlei": 7, "center": 7, "1557": 7, "21": 7, "anni": 7, "sauer": 7, "jame": 7, "warner": 7, "reliabl": 7, "transon": 7, "0632": 7, "22": 7, "thelen": 7, "d": 7, "bryson": 7, "b": 7, "k": 7, "beran": 7, "studi": 7, "23": 7, "dean": 7, "philip": 7, "dimension": 7, "algorithm": 7, "april": 7, "3390": 7, "a15040131": 7, "24": 7, "justin": 7, "pod": 7, "propulsor": 7, "august": 7, "2021": 7, "3032": 7, "25": 7, "layer": 7, "ingest": 7, "benefit": 7, "starc": 7, "abl": 7, "59": 7, "896": 7, "911": 7, "juli": 7, "c036103": 7, "26": 7, "chur": 7, "complement": 7, "eleventh": 7, "confer": 7, "fluid": [7, 9], "iccfd11": 7, "0702": 7, "url": 7, "www": 7, "iccfd": 7, "org": 7, "asset": 7, "pdf": 7, "iccfd11_pap": 7, "27": 7, "yildirm": 7, "robust": 7, "176459": 7, "scenarioaerodynam": 8, "nonlinearrunonc": [8, 10], "linearrunonc": [8, 10], "execut": [8, 9, 10], "pre": [8, 9, 10], "accept": [8, 9, 10], "valu": [8, 9, 10], "aero_build": [8, 9], "n": [8, 9, 10], "assembled_jac_typ": [8, 9, 10], "csc": [8, 9, 10], "dens": [8, 9, 10], "implicit": [8, 9, 10], "assembl": [8, 9, 10], "jacobian": [8, 9, 10], "auto_ord": [8, 9, 10], "fals": [8, 9, 10], "bool": [8, 9, 10], "depend": [8, 9, 10], "graph": [8, 9, 10], "It": [8, 9, 10], "break": [8, 9, 10], "reorder": [8, 9, 10], "cycl": [8, 9, 10], "multipointparallel": [8, 9, 10], "run_directori": [8, 9, 10], "path": [8, 9, 10], "empti": [8, 9, 10], "chang": [8, 9, 10], "directori": [8, 9, 10], "scenarioaerostructur": 9, "static": [9, 10], "project": 9, "geodisp": 9, "undeform": 9, "ti": 9, "togeth": 9, "principl": 9, "virtual": 9, "adjac": 9, "loop": 9, "tranfer": 9, "know": 9, "slice": 9, "shell": 9, "element": 9, "rotat": 9, "addit": 9, "translat": 9, "nonlinearblockg": 9, "linearblockg": 9, "use_aitken": 9, "coupling_group_typ": 9, "full_coupl": 9, "limit": 9, "flexibl": 9, "accomod": 9, "dlm": 9, "where": 9, "skip": 9, "aerodynamics_onli": 9, "ldxfer_build": 9, "post_coupling_ord": 9, "ldxfer": 9, "struct": 9, "pre_coupling_ord": 9, "struct_build": [9, 10], "scenariostructur": 10, "would": 10, "stress": 10}, "objects": {"mphys": [[0, 0, 0, "-", "builder"], [5, 0, 0, "-", "coupling_group"], [4, 0, 0, "-", "mphys_group"], [1, 0, 0, "-", "multipoint"], [5, 0, 0, "-", "scenario"]], "mphys.builder": [[0, 1, 1, "", "Builder"]], "mphys.builder.Builder": [[0, 2, 1, "", "get_coupling_group_subsystem"], [0, 2, 1, "", "get_mesh_coordinate_subsystem"], [0, 2, 1, "", "get_ndof"], [0, 2, 1, "", "get_number_of_nodes"], [0, 2, 1, "", "get_post_coupling_subsystem"], [0, 2, 1, "", "get_pre_coupling_subsystem"], [0, 2, 1, "", "get_tagged_indices"], [0, 2, 1, "", "initialize"]], "mphys.coupling_group": [[5, 1, 1, "", "CouplingGroup"]], "mphys.mphys_group": [[4, 1, 1, "", "MphysGroup"]], "mphys.mphys_group.MphysGroup": [[4, 2, 1, "", "configure"], [4, 2, 1, "", "mphys_add_subsystem"]], "mphys.multipoint": [[1, 1, 1, "", "Multipoint"], [1, 1, 1, "", "MultipointParallel"]], "mphys.multipoint.Multipoint": [[1, 2, 1, "", "mphys_add_scenario"], [1, 2, 1, "", "mphys_connect_scenario_coordinate_source"]], "mphys.multipoint.MultipointParallel": [[1, 2, 1, "", "mphys_add_scenario"]], "mphys.scenario": [[5, 1, 1, "", "Scenario"]], "mphys.scenario.Scenario": [[5, 2, 1, "", "initialize"], [5, 2, 1, "", "mphys_add_post_subsystem"], [5, 2, 1, "", "setup"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"]}, "titleterms": {"builder": [0, 9], "model": 1, "hierarchi": 1, "coupl": [1, 5], "group": [1, 5], "scenario": [1, 5, 6, 8, 9, 10], "multipoint": 1, "multipointparallel": 1, "variabl": [2, 4], "name": 2, "convent": 2, "tag": 3, "promot": 3, "The": 4, "mphysgroup": 4, "manual": 4, "connect": 4, "extend": 5, "librari": [5, 6], "initi": 5, "setup": 5, "document": 6, "mphy": [6, 7], "basic": [6, 8, 9, 10], "multiphys": 6, "singl": 6, "disciplin": 6, "develop": 6, "guid": 6, "refer": 6, "indic": 6, "tabl": 6, "paper": 7, "us": 7, "aerodynam": 8, "default": [8, 9, 10], "solver": [8, 9, 10], "option": [8, 9, 10], "n2": [8, 9, 10], "in_multipointparallel": [8, 9, 10], "geometry_build": [8, 9, 10], "aerostructur": 9, "requir": 9, "load": 9, "displac": 9, "transfer": 9, "structur": [9, 10]}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinxcontrib.bibtex": 9, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 58}, "alltitles": {"Builders": [[0, "builders"]], "Model Hierarchy": [[1, "model-hierarchy"]], "Coupling Groups": [[1, "coupling-groups"], [5, "coupling-groups"]], "Scenario Groups": [[1, "scenario-groups"]], "Multipoint Groups": [[1, "multipoint-groups"]], "Multipoint": [[1, "multipoint"]], "MultipointParallel": [[1, "multipointparallel"]], "Variable Naming Conventions": [[2, "variable-naming-conventions"]], "Tagged Promotion": [[3, "tagged-promotion"]], "The MphysGroup": [[4, "the-mphysgroup"]], "Manual Connection of Variables": [[4, "manual-connection-of-variables"]], "Extending the Scenario Library": [[5, "extending-the-scenario-library"]], "Scenarios": [[5, "scenarios"]], "Initialize": [[5, "initialize"]], "Setup": [[5, "setup"]], "Documentation for MPhys": [[6, "documentation-for-mphys"]], "MPhys Basics": [[6, "mphys-basics"], [6, null]], "MPhys Scenario Library": [[6, "mphys-scenario-library"]], "Multiphysics Scenarios": [[6, null]], "Single Discipline Scenarios": [[6, null]], "MPhys Developers Guide": [[6, "mphys-developers-guide"]], "Developers Guide": [[6, null]], "References": [[6, "references"]], "Indices and tables": [[6, "indices-and-tables"]], "Papers Using MPhys": [[7, "papers-using-mphys"]], "Aerodynamic Scenario": [[8, "aerodynamic-scenario"]], "Default Solvers": [[8, "default-solvers"], [9, "default-solvers"], [10, "default-solvers"]], "Options": [[8, "options"], [9, "options"], [10, "options"]], "N2:Basic": [[8, "n2-basic"], [10, "n2-basic"]], "N2: in_MultipointParallel": [[8, "n2-in-multipointparallel"], [9, "n2-in-multipointparallel"], [10, "n2-in-multipointparallel"]], "N2: in_MultipointParallel with geometry_builder": [[8, "n2-in-multipointparallel-with-geometry-builder"], [9, "n2-in-multipointparallel-with-geometry-builder"], [10, "n2-in-multipointparallel-with-geometry-builder"]], "Aerostructural Scenario": [[9, "aerostructural-scenario"]], "Builder Requirements": [[9, "builder-requirements"]], "Load and Displacement Transfer Builder": [[9, "load-and-displacement-transfer-builder"]], "Structural Solver Builder": [[9, "structural-solver-builder"]], "N2: Basic": [[9, "n2-basic"]], "Structural Scenario": [[10, "structural-scenario"]]}, "indexentries": {"builder (class in mphys.builder)": [[0, "mphys.builder.Builder"]], "get_coupling_group_subsystem() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_coupling_group_subsystem"]], "get_mesh_coordinate_subsystem() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_mesh_coordinate_subsystem"]], "get_ndof() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_ndof"]], "get_number_of_nodes() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_number_of_nodes"]], "get_post_coupling_subsystem() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_post_coupling_subsystem"]], "get_pre_coupling_subsystem() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_pre_coupling_subsystem"]], "get_tagged_indices() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_tagged_indices"]], "initialize() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.initialize"]], "module": [[0, "module-mphys.builder"], [1, "module-mphys.multipoint"], [4, "module-mphys.mphys_group"], [5, "module-mphys.coupling_group"], [5, "module-mphys.scenario"]], "mphys.builder": [[0, "module-mphys.builder"]], "multipoint (class in mphys.multipoint)": [[1, "mphys.multipoint.Multipoint"]], "multipointparallel (class in mphys.multipoint)": [[1, "mphys.multipoint.MultipointParallel"]], "mphys.multipoint": [[1, "module-mphys.multipoint"]], "mphys_add_scenario() (mphys.multipoint.multipoint method)": [[1, "mphys.multipoint.Multipoint.mphys_add_scenario"]], "mphys_add_scenario() (mphys.multipoint.multipointparallel method)": [[1, "mphys.multipoint.MultipointParallel.mphys_add_scenario"]], "mphys_connect_scenario_coordinate_source() (mphys.multipoint.multipoint method)": [[1, "mphys.multipoint.Multipoint.mphys_connect_scenario_coordinate_source"]], "mphysgroup (class in mphys.mphys_group)": [[4, "mphys.mphys_group.MphysGroup"]], "configure() (mphys.mphys_group.mphysgroup method)": [[4, "mphys.mphys_group.MphysGroup.configure"]], "mphys.mphys_group": [[4, "module-mphys.mphys_group"]], "mphys_add_subsystem() (mphys.mphys_group.mphysgroup method)": [[4, "mphys.mphys_group.MphysGroup.mphys_add_subsystem"]], "couplinggroup (class in mphys.coupling_group)": [[5, "mphys.coupling_group.CouplingGroup"]], "scenario (class in mphys.scenario)": [[5, "mphys.scenario.Scenario"]], "initialize() (mphys.scenario.scenario method)": [[5, "mphys.scenario.Scenario.initialize"]], "mphys.coupling_group": [[5, "module-mphys.coupling_group"]], "mphys.scenario": [[5, "module-mphys.scenario"]], "mphys_add_post_subsystem() (mphys.scenario.scenario method)": [[5, "mphys.scenario.Scenario.mphys_add_post_subsystem"]], "setup() (mphys.scenario.scenario method)": [[5, "mphys.scenario.Scenario.setup"]]}})