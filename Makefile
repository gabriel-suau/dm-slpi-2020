#########################################################
# 			USAGE				#
#							#
# 	- compilation en mode debug : make debug	#
# 	- compilation en mode optimisé : make release	#
#							#
#########################################################

# Compilateur + flags génériques
CC        = g++
CXX_FLAGS = -std=c++11 -I Eigen/Eigen

# Flags d'optimisation et de debug
# Le flag -Wall enclenche une myriade de warnings de Eigen sur lesquels nous n'avons pas le contrôle...
OPTIM_FLAGS = -O2 -DNDEBUG
DEBUG_FLAGS = -O0 -g -DDEBUG -pedantic

# Nom de l'exécutable
PROG = main
# Fichiers sources
SRC = main.cpp Solver.cpp MatrixUtils.cpp DataFile.cpp


# Mode release par défaut
.PHONY: release
release: CXX_FLAGS += $(OPTIM_FLAGS)
release: $(PROG)

# Mode debug
.PHONY: debug
debug: CXX_FLAGS += $(DEBUG_FLAGS)
debug: $(PROG)

# Compilation + édition de liens
$(PROG) : $(SRC)
	$(CC) $(SRC) $(CXX_FLAGS) -o $(PROG)

# Supprime l'exécutable, les fichiers binaires (.o) et les fichiers
# temporaires de sauvegarde (~)
clean :
	rm -f *.o *~ $(PROG)
