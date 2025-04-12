CXX ?= g++

BUILD_DIR ?= build
SRC_DIR ?= src

SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIR) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

SUM_TEST ?= false

CPPFLAGS ?= $(INC_FLAGS) -O2 -pedantic-errors -Wall -Wextra -Werror -MMD -MP

mult: $(OBJS)
	@echo "Linking: $@"
	@$(CXX) $(OBJS) -o $@ $(LDFLAGS)

sum: $(OBJS)
	@echo "Linking: $@"
	@$(CXX) $(OBJS) -o $@ -DSUM $(LDFLAGS)

# c source
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c 
	@$(MKDIR_P) $(dir $@)
	@echo "Compiling: $< -> $@"
	@$(CXX) $(CPPFLAGS) -c $< -o $@

# c++ source
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp 
	@$(MKDIR_P) $(dir $@)
	@echo "Compiling: $< -> $@"
	@$(CXX) $(CPPFLAGS) -c $< -o $@

.PHONY: test
sum-test: sum
	python3 scripts/test.py sum

mult-test: mult
	python3 scripts/test.py mult

.PHONY: clean
clean:
	@echo "Deleting build directory"
	@$(RM) -r $(BUILD_DIR)
	@echo "Deleting binary"
	@$(RM) $(BIN_NAME)

-include $(DEPS)

MKDIR_P ?= mkdir -p
