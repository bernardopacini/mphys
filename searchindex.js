Search.setIndex({"docnames": ["basics/builders", "basics/model_hierarchy", "basics/naming_conventions", "basics/remote_components", "basics/tagged_promotion", "developers/mphys_group", "developers/new_multiphysics_problems", "index", "references/papers_using_mphys", "scenarios/aerodynamic", "scenarios/aerostructural", "scenarios/structural"], "filenames": ["basics/builders.rst", "basics/model_hierarchy.rst", "basics/naming_conventions.rst", "basics/remote_components.rst", "basics/tagged_promotion.rst", "developers/mphys_group.rst", "developers/new_multiphysics_problems.rst", "index.rst", "references/papers_using_mphys.rst", "scenarios/aerodynamic.rst", "scenarios/aerostructural.rst", "scenarios/structural.rst"], "titles": ["Builders", "Model Hierarchy", "Variable Naming Conventions", "Remote Components", "Tagged Promotion", "The MphysGroup", "Extending the Scenario Library", "Documentation for MPhys", "Papers Using MPhys", "Aerodynamic Scenario", "Aerostructural Scenario", "Structural Scenario"], "terms": {"In": [0, 1, 3, 5, 6, 8], "larg": [0, 7], "multiphys": [0, 1, 2, 6], "problem": [0, 1, 2, 3, 6, 7, 10], "creation": 0, "connect": [0, 1, 4], "openmdao": [0, 1, 2, 3, 5, 7], "can": [0, 1, 3, 4, 5, 6, 10], "complic": 0, "time": [0, 3, 6], "consum": 0, "The": [0, 1, 3, 4, 6, 7, 9, 10, 11], "design": [0, 1, 3, 4, 8], "mphy": [0, 1, 2, 3, 4, 5, 6, 9, 10, 11], "i": [0, 1, 2, 3, 5, 6, 7, 9, 10, 11], "base": [0, 3, 5, 8, 9, 10, 11], "class": [0, 1, 3, 5, 6], "order": [0, 7, 9, 10, 11], "reduc": 0, "burden": 0, "user": [0, 1, 4, 6], "most": [0, 6], "assembli": 0, "model": [0, 3, 7, 8], "handl": [0, 8], "set": [0, 1, 2, 3, 5, 6, 7, 9, 10, 11], "helper": [0, 1], "object": [0, 1, 3, 4], "develop": [0, 6], "wish": [0, 4], "integr": 0, "code": [0, 2, 7], "should": [0, 1, 3, 5, 6], "subclass": [0, 1, 5, 6], "implement": [0, 3, 5, 6, 10], "method": [0, 1, 3, 4, 5, 6, 8, 10], "relev": [0, 3], "Not": 0, "all": [0, 3, 6], "need": [0, 1, 4, 5, 6, 9, 10, 11], "For": [0, 1, 10], "exampl": [0, 1, 10, 11], "transfer": [0, 1, 7, 8], "scheme": [0, 1], "mai": [0, 3], "precoupl": 0, "post": [0, 6, 9, 10, 11], "coupl": [0, 2, 4, 5, 7, 8, 9, 10, 11], "subsystem": [0, 1, 3, 4, 5, 6, 9, 10, 11], "scenario": [0, 3, 4, 5], "sourc": [0, 1, 3, 4, 5, 6], "templat": 0, "creat": [0, 1], "becaus": [0, 1, 5, 10], "mpi": [0, 1], "commun": [0, 1, 3], "us": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11], "insid": [0, 3, 9, 10, 11], "known": 0, "when": [0, 2, 3], "instanti": [0, 1, 3, 6], "actual": 0, "solver": [0, 1, 2, 5, 6, 7, 8], "etc": 0, "constructor": 0, "initi": [0, 1, 3], "comm": [0, 1, 3], "thi": [0, 1, 2, 3, 4, 6, 7, 9, 10, 11], "call": [0, 1, 3, 5, 6], "avail": [0, 3, 7], "paramet": [0, 1, 3, 6], "xfer": 0, "instanc": [0, 3, 5], "get_mesh_coordinate_subsystem": 0, "scenario_nam": [0, 4], "none": [0, 1, 3, 6, 10], "contain": [0, 1, 3], "return": [0, 10], "mesh": [0, 1, 4, 10], "coordin": [0, 1, 2, 4, 10], "str": [0, 1, 3, 6, 9, 10, 11], "name": [0, 1, 3, 5, 6, 7], "compon": [0, 1, 4, 6, 7, 8, 9, 10, 11], "group": [0, 3, 5, 7, 9, 10, 11], "ha": [0, 1, 3, 7], "an": [0, 1, 3, 4, 5, 6, 8, 9, 10, 11], "output": [0, 3, 4, 10], "get_coupling_group_subsystem": [0, 6, 10], "add": [0, 1, 3, 5, 6, 7, 10], "couplinggroup": [0, 1, 4, 5, 6], "comput": [0, 1, 3, 4, 8, 10], "multipl": [0, 3], "get_pre_coupling_subsystem": 0, "ad": [0, 1, 3, 5, 6, 9, 10, 11], "each": [0, 1, 3, 6, 7, 10], "befor": [0, 1, 3], "get_post_coupling_subsystem": 0, "after": [0, 1, 3], "get_number_of_nod": 0, "number": [0, 1, 2, 3], "node": [0, 3, 10], "defin": [0, 3, 6], "interfac": [0, 2, 7], "input": [0, 1, 3, 4, 6], "number_of_nod": 0, "int": [0, 3], "domain": [0, 8], "get_ndof": [0, 10], "degre": [0, 10], "freedom": [0, 10], "locat": 0, "ndof": 0, "get_tagged_indic": 0, "tag": [0, 1, 2, 5, 6, 7], "grid": 0, "id": 0, "list": [0, 1, 3], "bodi": 0, "boundari": [0, 8], "integ": 0, "string": [0, 9, 10, 11], "grid_id": 0, "correspond": 0, "given": [0, 1, 3, 7, 10], "pattern": [1, 6], "build": 1, "optim": [1, 3, 8], "level": [1, 3, 4], "differ": [1, 2, 3, 4], "type": [1, 3, 4, 6, 7, 9, 10, 11], "provid": [1, 2, 3, 7], "highest": 1, "consist": 1, "which": [1, 3, 9, 10, 11], "repres": [1, 4, 6], "condit": [1, 6], "analys": [1, 3], "perform": [1, 3, 6], "within": [1, 3], "primari": [1, 10], "builder": [1, 6, 7, 9, 11], "ar": [1, 3, 4, 5, 7, 9, 10, 11], "help": [1, 2, 6], "popul": 1, "from": [1, 3, 4, 5, 6, 10, 11], "promot": [1, 5, 6, 7], "specif": [1, 6], "variabl": [1, 3, 4, 6, 7, 8], "physic": [1, 2, 3, 7, 10], "being": [1, 3, 7], "solv": [1, 2, 7], "That": 1, "modul": [1, 7, 10], "aerodynam": [1, 2, 7, 8, 10], "structur": [1, 2, 7, 8], "potenti": [1, 8], "interpol": 1, "between": [1, 3], "load": [1, 3, 11], "displac": [1, 2], "typic": [1, 10], "associ": [1, 2, 3, 6], "automat": [1, 5, 9, 10, 11], "proper": 1, "have": [1, 3, 5, 6], "default": [1, 3, 5, 6], "nonlinear": [1, 3, 5, 6, 8], "linear": [1, 3, 5, 6, 8, 9, 10, 11], "overwritten": 1, "option": [1, 6, 7], "argument": [1, 3, 6], "mphys_add_scenario": 1, "could": 1, "cruis": 1, "flight": 1, "requir": [1, 2, 3, 6, 7, 9, 11], "determin": [1, 3, 9, 10, 11], "lift": [1, 8], "drag": 1, "ani": [1, 3, 4, 5, 6], "occur": 1, "sonic": 1, "boom": 1, "propag": 1, "flow": [1, 2], "solut": [1, 3], "one": [1, 3, 6], "wai": 1, "doe": [1, 7], "therefor": 1, "put": 1, "converg": 1, "librari": 1, "see": 1, "detail": [1, 7], "about": [1, 10], "standard": [1, 5, 7], "If": [1, 3, 5, 6, 9, 10, 11], "particular": [1, 2, 3, 4, 6], "cover": 1, "new": [1, 3, 5, 7], "mphysgroup": [1, 7], "There": [1, 4], "two": [1, 3], "version": 1, "deriv": [1, 3], "parallelgroup": 1, "both": [1, 3, 4, 10], "function": [1, 3], "lower": 1, "sequenti": 1, "evalu": [1, 3], "top": [1, 3], "setup": [1, 3, 5], "follow": [1, 3, 7], "step": 1, "must": [1, 3, 6, 10], "done": [1, 3, 5], "": [1, 3, 5, 6, 8, 9, 10, 11], "self": [1, 6], "other": [1, 3, 4, 6], "like": [1, 4, 6], "geometri": [1, 9, 10, 11], "addition": 1, "hold": 1, "These": [1, 4, 7], "extra": 1, "kwarg": [1, 3, 5, 6], "extens": [1, 7], "block": [1, 5, 6], "gauss": [1, 5, 6], "seidel": [1, 5, 6], "coupling_nonlinear_solv": 1, "coupling_linear_solv": 1, "nonlinearsolv": 1, "assign": [1, 3], "primal": 1, "linearsolv": 1, "sensit": 1, "mphys_connect_scenario_coordinate_sourc": 1, "disciplin": [1, 4], "A": [1, 3, 5, 6, 8, 9, 10, 11], "aid": 1, "target": 1, "assum": [1, 3], "x_": 1, "0": [1, 3], "api": 1, "rank": [1, 3], "greater": 1, "than": [1, 3, 5], "equal": 1, "simultan": 1, "unlik": 1, "so": [1, 3], "in_multipointparallel": [1, 6], "true": [1, 3, 9, 10, 11], "outsid": [1, 4], "cannot": 1, "directli": [1, 5, 6], "higher": 1, "parallel": [1, 3, 7], "mpi_proc_alloc": 1, "while": [2, 3, 5, 7], "possibl": 2, "up": [2, 7], "same": [2, 3], "prefer": 2, "more": [2, 3, 7], "easili": 2, "interchang": 2, "tabl": 2, "descript": [2, 3, 7, 9, 10, 11], "x_aero0": 2, "mphys_coordin": [2, 4], "surfac": [2, 8, 10], "jig": [2, 8, 10], "shape": [2, 8, 10], "x_aero": 2, "mphys_coupl": [2, 4], "deform": 2, "u_aero": 2, "f_aero": 2, "forc": [2, 3, 10], "t_convect": 2, "temperatur": 2, "convect": 2, "q_convect": 2, "heat": [2, 8], "x_struct0": 2, "u_struct": 2, "state": [2, 3, 4, 10], "vector": [2, 3, 10], "f_struct": 2, "t_conduct": 2, "thermal": 2, "side": [2, 3], "q_conduct": 2, "conduct": 2, "To": [2, 3, 6], "make": [2, 3, 6], "swap": 2, "easier": [2, 3], "also": [2, 6], "share": [2, 4], "noncoupl": 2, "aoa": 2, "mphys_input": [2, 4], "angl": 2, "attack": 2, "pleas": 2, "includ": [2, 3, 6, 8, 10], "unit": 2, "deg": 2, "rad": 2, "declar": [2, 3, 6], "yaw": 2, "mach": 2, "refer": 2, "reynold": 2, "q_inf": 2, "dynam": [2, 8], "pressur": 2, "purpos": [3, 5], "mean": 3, "analysi": [3, 6, 7, 8], "local": 3, "One": 3, "situat": 3, "desir": 3, "carri": 3, "out": 3, "full": [3, 8, 10], "exce": 3, "hpc": 3, "job": 3, "Such": 3, "without": 3, "normal": 3, "manual": [3, 4], "restart": 3, "would": [3, 11], "thu": 3, "capabl": [3, 7], "keep": 3, "serial": 3, "run": [3, 6], "continu": 3, "login": 3, "e": [3, 8], "g": 3, "nohup": 3, "screen": 3, "linux": 3, "command": 3, "across": [3, 7], "sever": 3, "anoth": 3, "where": [3, 10], "advantag": 3, "streamlin": 3, "massiv": 3, "environ": 3, "gener": [3, 7, 11], "nest": 3, "server": 3, "client": 3, "arrang": 3, "outer": 3, "serv": 3, "overarch": 3, "inner": 3, "isol": 3, "high": [3, 7, 8], "fidel": [3, 7, 8], "remain": 3, "open": 3, "gradient": [3, 8], "wall": 3, "save": 3, "maximum": 3, "previou": 3, "multipli": 3, "scale": 3, "factor": 3, "relaunch": 3, "three": 3, "achiev": 3, "remotecomp": 3, "explicit": 3, "wrap": 3, "replic": 3, "request": 3, "estim": 3, "servermanag": 3, "control": 3, "pbs4py": 3, "zeromq": 3, "network": 3, "through": 3, "mphyszeromqservermanag": 3, "encod": 3, "json": 3, "dictionari": 3, "send": 3, "receiv": 3, "necessari": [3, 6, 7], "inform": 3, "socket": 3, "ssh": 3, "port": 3, "forward": 3, "start": 3, "stop": 3, "check": 3, "statu": 3, "mphyszeromqserv": 3, "accept": [3, 9, 10, 11], "valu": [3, 9, 10, 11], "acceptable_port_rang": 3, "5081": 3, "6000": 3, "n": [3, 9, 10, 11], "rang": 3, "look": 3, "busi": 3, "additional_remote_input": 3, "addit": [3, 10], "var": 3, "additional_remote_output": 3, "constraint": [3, 4, 8, 11], "additional_server_arg": 3, "give": 3, "always_opt": 3, "fals": [3, 9, 10, 11], "bool": [3, 9, 10, 11], "oper": [3, 6], "loop": [3, 10], "even": 3, "respons": 3, "distribut": [3, 8], "process": 3, "dump_json": 3, "dump": 3, "file": 3, "dump_separate_json": 3, "separ": 3, "pb": 3, "launcher": 3, "reboot_only_on_function_cal": 3, "onli": [3, 4, 9, 11], "allow": 3, "reboot": 3, "avoid": 3, "rerun": 3, "next": 3, "shorten": 3, "run_root_onli": 3, "compute_parti": 3, "apply_linear": 3, "apply_nonlinear": 3, "compute_jacvec_product": 3, "broadcast": 3, "result": [3, 4], "run_server_filenam": 3, "mphys_serv": 3, "py": 3, "python": 3, "launch": 3, "time_estimate_buff": 3, "float": 3, "constant": 3, "second": 3, "esim": 3, "veri": [3, 7], "slowest": 3, "faster": 3, "expir": 3, "slower": 3, "time_estimate_multipli": 3, "2": [3, 8, 10], "whether": 3, "max": 3, "prior": 3, "use_derivative_color": 3, "color": 3, "case": [3, 6, 11], "var_naming_dot_replac": 3, "what": 3, "replac": 3, "dv": 3, "tree": 3, "facilit": 3, "get_om_group_function_point": 3, "pointer": 3, "multipoint": [3, 6, 8], "By": 3, "On": 3, "charact": 3, "nth": 3, "sent": 3, "mphys_": [3, 6], "_servern": 3, "search": [3, 7], "keyword": 3, "displai": 3, "do": [3, 6], "submiss": 3, "script": 3, "note": 3, "support": 3, "systemerror": 3, "trigger": 3, "otherwis": [3, 6], "superson": 3, "panel": 3, "aerostructur": [3, 7, 8], "as_opt_remote_seri": 3, "as_opt_remote_parallel": 3, "as_opt_parallel": 3, "multipointparallel": [3, 9, 10, 11], "hand": 3, "point": [3, 6], "either": 3, "filenam": 3, "As": [3, 6], "demonstr": 3, "configur": [3, 4, 5, 6, 8], "functor": 3, "getmodel": 3, "combin": [3, 8], "run_directori": [3, 9, 10, 11], "directori": [3, 9, 10, 11], "k4": 3, "monitor": 3, "queue": 3, "nasa": [3, 8], "k": [3, 8], "cluster": 3, "write": 3, "data": [3, 4], "except": 3, "wall_tim": 3, "entri": 3, "complet": 3, "design_count": 3, "track": 3, "how": [3, 7], "mani": 3, "been": 3, "written": 3, "n2": 3, "titl": 3, "n2_inner_analysis_": 3, "html": 3, "your": [3, 5, 6], "stop_serv": 3, "manag": 3, "background": 3, "unexpectedli": 3, "difficult": 3, "involv": 3, "find": 3, "wa": 3, "somewhat": 3, "howev": [3, 7], "well": 3, "wrt": 3, "check_tot": 3, "compute_tot": 3, "depend": [3, 9, 10, 11], "costli": 3, "remote_compon": 3, "_setup_server_manag": 3, "_send_inputs_to_serv": 3, "_receive_outputs_from_serv": 3, "store": 3, "some": [3, 5], "bound": 3, "we": 3, "detect": 3, "runtim": 3, "overrid": 3, "attribut": 3, "pathnam": 3, "unscal": 3, "dimension": [3, 8], "read": 3, "via": 3, "kei": 3, "discrete_input": 3, "dict": 3, "discret": 3, "discrete_output": 3, "partial": 3, "sub": 3, "jacobian": [3, 9, 10, 11], "part": 3, "jac": 3, "output_nam": 3, "input_nam": [3, 4], "server_manag": 3, "start_serv": 3, "enough_time_is_remain": 3, "estimated_model_tim": 3, "enough": 3, "much": 3, "take": 3, "ignore_setup_warn": 3, "ignore_runtime_warn": 3, "rerun_initial_design": 3, "await": 3, "back": 3, "_parse_incoming_messag": 3, "_send_outputs_to_cli": 3, "ignor": 3, "warn": 3, "baselin": 3, "upon": 3, "starup": 3, "zmq_pb": 3, "component_nam": 3, "captur": 3, "_server": 3, "server_numb": 3, "altern": 3, "specifi": 3, "alreadi": 3, "select": 4, "four": 4, "among": 4, "come": 4, "mphys_result": 4, "you": [4, 5, 6], "result_nam": 4, "access": 4, "thei": [4, 7], "further": 4, "behind": 4, "curtain": 4, "autom": [4, 5], "mechanan": 5, "mphys_add_subsystem": [5, 6], "still": 5, "add_subsystem": 5, "dure": 5, "phase": [5, 6], "sure": 5, "parent": [5, 6], "super": [5, 6], "import": 5, "understand": [5, 8], "interact": [5, 10], "inherit": 5, "rather": 5, "mphys_group": 5, "appropri": 5, "get": 6, "them": 6, "balanc": 6, "suitabl": 6, "coupling_group": 6, "custom": 6, "least": 6, "mode": 6, "free": 6, "basic": 6, "mphys_add_pre_coupling_subsystem": 6, "mphys_add_post_coupling_subsystem": 6, "main": 6, "_mphys_scenario_setup": 6, "mphys_add_post_subsystem": 6, "promotes_input": 6, "promotes_output": 6, "end": 6, "system": [6, 8], "iter": 6, "tupl": [6, 10], "els": 6, "packag": [7, 8], "eas": 7, "straightforward": 7, "multidisciplinari": [7, 8], "address": 7, "its": 7, "convent": 7, "absolut": 7, "guidelin": 7, "usag": 7, "modular": 7, "technologi": 7, "collabor": 7, "area": 7, "research": [7, 8], "strive": 7, "work": [7, 10], "hierarchi": 7, "remot": 7, "describ": 7, "page": 7, "extend": 7, "paper": 7, "index": 7, "1": [8, 10], "m": 8, "saja": 8, "abdul": 8, "kaiyoom": 8, "anil": 8, "yildirim": 8, "joaquim": 8, "r": 8, "martin": 8, "aeropropuls": 8, "over": 8, "wing": 8, "nacel": 8, "aiaa": 8, "scitech": 8, "forum": 8, "januari": 8, "2023": 8, "doi": 8, "10": 8, "2514": 8, "6": 8, "0327": 8, "ran": 8, "aviat": 8, "san": 8, "diego": 8, "ca": 8, "june": 8, "3588": 8, "3": [8, 10], "josh": 8, "l": 8, "anib": 8, "charl": 8, "mader": 8, "aerotherm": 8, "x": 8, "57": 8, "motor": 8, "orlando": 8, "fl": 8, "2020": 8, "2115": 8, "4": [8, 10], "cfd": 8, "plate": 8, "fin": 8, "exchang": 8, "2022": 8, "3930": 8, "5": 8, "joshua": 8, "phd": 8, "thesi": 8, "univers": 8, "michigan": 8, "ann": 8, "arbor": 8, "http": 8, "deepblu": 8, "lib": 8, "umich": 8, "edu": 8, "2027": 8, "42": 8, "171375": 8, "electr": 8, "aircraft": 8, "conjug": 8, "intern": 8, "journal": 8, "mass": 8, "189": 8, "122689": 8, "1016": 8, "j": 8, "ijheatmasstransf": 8, "7": 8, "garo": 8, "bedonian": 8, "jason": 8, "hicken": 8, "adapt": 8, "sampl": 8, "enhanc": 8, "surrog": 8, "american": 8, "institut": 8, "aeronaut": 8, "astronaut": 8, "3998": 8, "8": 8, "adrien": 8, "crovato": 8, "romain": 8, "boman": 8, "vincent": 8, "terrapon": 8, "grigorio": 8, "dimitriadi": 8, "alex": 8, "p": 8, "prado": 8, "pedro": 8, "h": 8, "cabral": 8, "fast": 8, "calcul": 8, "preliminari": 8, "aeroelast": 8, "9": 8, "alasdair": 8, "c": 8, "grai": 8, "graem": 8, "kennedi": 8, "geometr": 8, "3316": 8, "hannah": 8, "hajdik": 8, "bernardo": 8, "pacini": 8, "benjamin": 8, "brelj": 8, "3589": 8, "11": 8, "ping": 8, "he": 8, "heyecan": 8, "koyuncuoglu": 8, "helen": 8, "hu": 8, "anvesh": 8, "dhulipalla": 8, "haiyang": 8, "hui": 8, "uav": 8, "propel": 8, "adjoint": 8, "0531": 8, "12": 8, "kevin": 8, "jacobson": 8, "bret": 8, "stanford": 8, "flutter": [8, 10], "constrain": 8, "frequenc": 8, "approach": 8, "2242": 8, "13": 8, "multi": 8, "1844": 8, "14": 8, "andrew": 8, "lamkin": 8, "nathan": 8, "wuki": 8, "advanc": 8, "bypass": 8, "turbofan": 8, "engin": 8, "3591": 8, "15": 8, "33rd": 8, "congress": 8, "council": 8, "scienc": 8, "septemb": 8, "16": 8, "christoph": 8, "lupp": 8, "inclus": 8, "effect": 8, "163259": 8, "17": 8, "malhar": 8, "prajapati": 8, "karthik": 8, "duraisami": 8, "urban": 8, "air": 8, "mobil": 8, "vehicl": 8, "0326": 8, "18": 8, "toward": 8, "mix": 8, "aero": [8, 10], "acoust": 8, "3905": 8, "19": 8, "propuls": 8, "tiltw": 8, "concept": 8, "0143": 8, "20": 8, "pawel": 8, "chwalowski": 8, "ongo": 8, "predict": 8, "valid": 8, "activ": 8, "langlei": 8, "center": 8, "1557": 8, "21": 8, "anni": 8, "sauer": 8, "jame": 8, "warner": 8, "reliabl": 8, "transon": 8, "0632": 8, "22": 8, "thelen": 8, "d": 8, "bryson": 8, "b": 8, "beran": 8, "studi": 8, "23": 8, "dean": 8, "philip": 8, "algorithm": 8, "april": 8, "3390": 8, "a15040131": 8, "24": 8, "justin": 8, "pod": 8, "propulsor": 8, "august": 8, "2021": 8, "3032": 8, "25": 8, "layer": 8, "ingest": 8, "benefit": 8, "starc": 8, "abl": 8, "59": 8, "896": 8, "911": 8, "juli": 8, "c036103": 8, "26": 8, "chur": 8, "complement": 8, "eleventh": 8, "confer": 8, "fluid": [8, 10], "iccfd11": 8, "0702": 8, "url": 8, "www": 8, "iccfd": 8, "org": 8, "asset": 8, "pdf": 8, "iccfd11_pap": 8, "27": 8, "yildirm": 8, "robust": 8, "176459": 8, "scenarioaerodynam": 9, "nonlinearrunonc": [9, 11], "linearrunonc": [9, 11], "execut": [9, 10, 11], "pre": [9, 10, 11], "aero_build": [9, 10], "assembled_jac_typ": [9, 10, 11], "csc": [9, 10, 11], "dens": [9, 10, 11], "implicit": [9, 10, 11], "assembl": [9, 10, 11], "auto_ord": [9, 10, 11], "graph": [9, 10, 11], "It": [9, 10, 11], "break": [9, 10, 11], "reorder": [9, 10, 11], "cycl": [9, 10, 11], "path": [9, 10, 11], "empti": [9, 10, 11], "chang": [9, 10, 11], "scenarioaerostructur": 10, "static": [10, 11], "project": 10, "geodisp": 10, "undeform": 10, "ti": 10, "togeth": 10, "principl": 10, "virtual": 10, "adjac": 10, "tranfer": 10, "know": 10, "slice": 10, "shell": 10, "element": 10, "rotat": 10, "translat": 10, "nonlinearblockg": 10, "linearblockg": 10, "use_aitken": 10, "coupling_group_typ": 10, "full_coupl": 10, "limit": 10, "flexibl": 10, "accomod": 10, "dlm": 10, "skip": 10, "aerodynamics_onli": 10, "ldxfer_build": 10, "post_coupling_ord": 10, "ldxfer": 10, "struct": 10, "pre_coupling_ord": 10, "struct_build": [10, 11], "scenariostructur": 11, "stress": 11}, "objects": {"mphys": [[0, 0, 0, "-", "builder"], [6, 0, 0, "-", "coupling_group"], [5, 0, 0, "-", "mphys_group"], [1, 0, 0, "-", "multipoint"], [6, 0, 0, "-", "scenario"]], "mphys.builder": [[0, 1, 1, "", "Builder"]], "mphys.builder.Builder": [[0, 2, 1, "", "get_coupling_group_subsystem"], [0, 2, 1, "", "get_mesh_coordinate_subsystem"], [0, 2, 1, "", "get_ndof"], [0, 2, 1, "", "get_number_of_nodes"], [0, 2, 1, "", "get_post_coupling_subsystem"], [0, 2, 1, "", "get_pre_coupling_subsystem"], [0, 2, 1, "", "get_tagged_indices"], [0, 2, 1, "", "initialize"]], "mphys.coupling_group": [[6, 1, 1, "", "CouplingGroup"]], "mphys.mphys_group": [[5, 1, 1, "", "MphysGroup"]], "mphys.mphys_group.MphysGroup": [[5, 2, 1, "", "configure"], [5, 2, 1, "", "mphys_add_subsystem"]], "mphys.multipoint": [[1, 1, 1, "", "Multipoint"], [1, 1, 1, "", "MultipointParallel"]], "mphys.multipoint.Multipoint": [[1, 2, 1, "", "mphys_add_scenario"], [1, 2, 1, "", "mphys_connect_scenario_coordinate_source"]], "mphys.multipoint.MultipointParallel": [[1, 2, 1, "", "mphys_add_scenario"]], "mphys.network.remote_component": [[3, 1, 1, "", "RemoteComp"]], "mphys.network.remote_component.RemoteComp": [[3, 2, 1, "", "compute"], [3, 2, 1, "", "compute_partials"], [3, 2, 1, "", "initialize"], [3, 2, 1, "", "setup"]], "mphys.network.server": [[3, 1, 1, "", "Server"]], "mphys.network.server.Server": [[3, 2, 1, "", "run"]], "mphys.network.server_manager": [[3, 1, 1, "", "ServerManager"]], "mphys.network.server_manager.ServerManager": [[3, 2, 1, "", "enough_time_is_remaining"], [3, 2, 1, "", "start_server"], [3, 2, 1, "", "stop_server"]], "mphys.network.zmq_pbs": [[3, 1, 1, "", "MPhysZeroMQServer"], [3, 1, 1, "", "MPhysZeroMQServerManager"], [3, 1, 1, "", "RemoteZeroMQComp"]], "mphys.network.zmq_pbs.MPhysZeroMQServerManager": [[3, 2, 1, "", "enough_time_is_remaining"], [3, 2, 1, "", "start_server"], [3, 2, 1, "", "stop_server"]], "mphys.network.zmq_pbs.RemoteZeroMQComp": [[3, 2, 1, "", "initialize"]], "mphys.scenario": [[6, 1, 1, "", "Scenario"]], "mphys.scenario.Scenario": [[6, 2, 1, "", "initialize"], [6, 2, 1, "", "mphys_add_post_subsystem"], [6, 2, 1, "", "setup"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"]}, "titleterms": {"builder": [0, 10], "model": 1, "hierarchi": 1, "coupl": [1, 6], "group": [1, 6], "scenario": [1, 6, 7, 9, 10, 11], "multipoint": 1, "multipointparallel": 1, "variabl": [2, 5], "name": 2, "convent": 2, "remot": 3, "compon": 3, "remotezeromqcomp": 3, "option": [3, 9, 10, 11], "usag": 3, "exampl": 3, "troubleshoot": 3, "current": 3, "limit": 3, "tag": 4, "promot": 4, "The": 5, "mphysgroup": 5, "manual": 5, "connect": 5, "extend": 6, "librari": [6, 7], "initi": 6, "setup": 6, "document": 7, "mphy": [7, 8], "basic": [7, 9, 10, 11], "multiphys": 7, "singl": 7, "disciplin": 7, "develop": 7, "guid": 7, "refer": 7, "indic": 7, "tabl": 7, "paper": 8, "us": 8, "aerodynam": 9, "default": [9, 10, 11], "solver": [9, 10, 11], "n2": [9, 10, 11], "in_multipointparallel": [9, 10, 11], "geometry_build": [9, 10, 11], "aerostructur": 10, "requir": 10, "load": 10, "displac": 10, "transfer": 10, "structur": [10, 11]}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinxcontrib.bibtex": 9, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 58}, "alltitles": {"Builders": [[0, "builders"]], "Model Hierarchy": [[1, "model-hierarchy"]], "Coupling Groups": [[1, "coupling-groups"], [6, "coupling-groups"]], "Scenario Groups": [[1, "scenario-groups"]], "Multipoint Groups": [[1, "multipoint-groups"]], "Multipoint": [[1, "multipoint"]], "MultipointParallel": [[1, "multipointparallel"]], "Variable Naming Conventions": [[2, "variable-naming-conventions"]], "Remote Components": [[3, "remote-components"]], "RemoteZeroMQComp Options": [[3, "remotezeromqcomp-options"]], "Usage": [[3, "usage"]], "Example": [[3, "example"]], "Troubleshooting": [[3, "troubleshooting"]], "Current Limitations": [[3, "current-limitations"]], "Tagged Promotion": [[4, "tagged-promotion"]], "The MphysGroup": [[5, "the-mphysgroup"]], "Manual Connection of Variables": [[5, "manual-connection-of-variables"]], "Extending the Scenario Library": [[6, "extending-the-scenario-library"]], "Scenarios": [[6, "scenarios"]], "Initialize": [[6, "initialize"]], "Setup": [[6, "setup"]], "Documentation for MPhys": [[7, "documentation-for-mphys"]], "MPhys Basics": [[7, "mphys-basics"], [7, null]], "MPhys Scenario Library": [[7, "mphys-scenario-library"]], "Multiphysics Scenarios": [[7, null]], "Single Discipline Scenarios": [[7, null]], "MPhys Developers Guide": [[7, "mphys-developers-guide"]], "Developers Guide": [[7, null]], "References": [[7, "references"]], "Indices and tables": [[7, "indices-and-tables"]], "Papers Using MPhys": [[8, "papers-using-mphys"]], "Aerodynamic Scenario": [[9, "aerodynamic-scenario"]], "Default Solvers": [[9, "default-solvers"], [10, "default-solvers"], [11, "default-solvers"]], "Options": [[9, "options"], [10, "options"], [11, "options"]], "N2:Basic": [[9, "n2-basic"], [11, "n2-basic"]], "N2: in_MultipointParallel": [[9, "n2-in-multipointparallel"], [10, "n2-in-multipointparallel"], [11, "n2-in-multipointparallel"]], "N2: in_MultipointParallel with geometry_builder": [[9, "n2-in-multipointparallel-with-geometry-builder"], [10, "n2-in-multipointparallel-with-geometry-builder"], [11, "n2-in-multipointparallel-with-geometry-builder"]], "Aerostructural Scenario": [[10, "aerostructural-scenario"]], "Builder Requirements": [[10, "builder-requirements"]], "Load and Displacement Transfer Builder": [[10, "load-and-displacement-transfer-builder"]], "Structural Solver Builder": [[10, "structural-solver-builder"]], "N2: Basic": [[10, "n2-basic"]], "Structural Scenario": [[11, "structural-scenario"]]}, "indexentries": {"builder (class in mphys.builder)": [[0, "mphys.builder.Builder"]], "get_coupling_group_subsystem() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_coupling_group_subsystem"]], "get_mesh_coordinate_subsystem() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_mesh_coordinate_subsystem"]], "get_ndof() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_ndof"]], "get_number_of_nodes() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_number_of_nodes"]], "get_post_coupling_subsystem() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_post_coupling_subsystem"]], "get_pre_coupling_subsystem() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_pre_coupling_subsystem"]], "get_tagged_indices() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.get_tagged_indices"]], "initialize() (mphys.builder.builder method)": [[0, "mphys.builder.Builder.initialize"]], "module": [[0, "module-mphys.builder"], [1, "module-mphys.multipoint"], [5, "module-mphys.mphys_group"], [6, "module-mphys.coupling_group"], [6, "module-mphys.scenario"]], "mphys.builder": [[0, "module-mphys.builder"]], "multipoint (class in mphys.multipoint)": [[1, "mphys.multipoint.Multipoint"]], "multipointparallel (class in mphys.multipoint)": [[1, "mphys.multipoint.MultipointParallel"]], "mphys.multipoint": [[1, "module-mphys.multipoint"]], "mphys_add_scenario() (mphys.multipoint.multipoint method)": [[1, "mphys.multipoint.Multipoint.mphys_add_scenario"]], "mphys_add_scenario() (mphys.multipoint.multipointparallel method)": [[1, "mphys.multipoint.MultipointParallel.mphys_add_scenario"]], "mphys_connect_scenario_coordinate_source() (mphys.multipoint.multipoint method)": [[1, "mphys.multipoint.Multipoint.mphys_connect_scenario_coordinate_source"]], "mphyszeromqserver (class in mphys.network.zmq_pbs)": [[3, "mphys.network.zmq_pbs.MPhysZeroMQServer"]], "mphyszeromqservermanager (class in mphys.network.zmq_pbs)": [[3, "mphys.network.zmq_pbs.MPhysZeroMQServerManager"]], "remotecomp (class in mphys.network.remote_component)": [[3, "mphys.network.remote_component.RemoteComp"]], "remotezeromqcomp (class in mphys.network.zmq_pbs)": [[3, "mphys.network.zmq_pbs.RemoteZeroMQComp"]], "server (class in mphys.network.server)": [[3, "mphys.network.server.Server"]], "servermanager (class in mphys.network.server_manager)": [[3, "mphys.network.server_manager.ServerManager"]], "compute() (mphys.network.remote_component.remotecomp method)": [[3, "mphys.network.remote_component.RemoteComp.compute"]], "compute_partials() (mphys.network.remote_component.remotecomp method)": [[3, "mphys.network.remote_component.RemoteComp.compute_partials"]], "enough_time_is_remaining() (mphys.network.server_manager.servermanager method)": [[3, "mphys.network.server_manager.ServerManager.enough_time_is_remaining"]], "enough_time_is_remaining() (mphys.network.zmq_pbs.mphyszeromqservermanager method)": [[3, "mphys.network.zmq_pbs.MPhysZeroMQServerManager.enough_time_is_remaining"]], "initialize() (mphys.network.remote_component.remotecomp method)": [[3, "mphys.network.remote_component.RemoteComp.initialize"]], "initialize() (mphys.network.zmq_pbs.remotezeromqcomp method)": [[3, "mphys.network.zmq_pbs.RemoteZeroMQComp.initialize"]], "run() (mphys.network.server.server method)": [[3, "mphys.network.server.Server.run"]], "setup() (mphys.network.remote_component.remotecomp method)": [[3, "mphys.network.remote_component.RemoteComp.setup"]], "start_server() (mphys.network.server_manager.servermanager method)": [[3, "mphys.network.server_manager.ServerManager.start_server"]], "start_server() (mphys.network.zmq_pbs.mphyszeromqservermanager method)": [[3, "mphys.network.zmq_pbs.MPhysZeroMQServerManager.start_server"]], "stop_server() (mphys.network.server_manager.servermanager method)": [[3, "mphys.network.server_manager.ServerManager.stop_server"]], "stop_server() (mphys.network.zmq_pbs.mphyszeromqservermanager method)": [[3, "mphys.network.zmq_pbs.MPhysZeroMQServerManager.stop_server"]], "mphysgroup (class in mphys.mphys_group)": [[5, "mphys.mphys_group.MphysGroup"]], "configure() (mphys.mphys_group.mphysgroup method)": [[5, "mphys.mphys_group.MphysGroup.configure"]], "mphys.mphys_group": [[5, "module-mphys.mphys_group"]], "mphys_add_subsystem() (mphys.mphys_group.mphysgroup method)": [[5, "mphys.mphys_group.MphysGroup.mphys_add_subsystem"]], "couplinggroup (class in mphys.coupling_group)": [[6, "mphys.coupling_group.CouplingGroup"]], "scenario (class in mphys.scenario)": [[6, "mphys.scenario.Scenario"]], "initialize() (mphys.scenario.scenario method)": [[6, "mphys.scenario.Scenario.initialize"]], "mphys.coupling_group": [[6, "module-mphys.coupling_group"]], "mphys.scenario": [[6, "module-mphys.scenario"]], "mphys_add_post_subsystem() (mphys.scenario.scenario method)": [[6, "mphys.scenario.Scenario.mphys_add_post_subsystem"]], "setup() (mphys.scenario.scenario method)": [[6, "mphys.scenario.Scenario.setup"]]}})