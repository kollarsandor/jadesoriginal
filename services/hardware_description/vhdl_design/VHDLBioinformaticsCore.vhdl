-- JADED Platform - VHDL Bioinformatics Hardware Core
-- Complete VHDL implementation for computational biology acceleration
-- Production-ready synthesizable code for FPGA and ASIC deployment

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.std_logic_unsigned.all;

-- Custom types and constants package
package bioinformatics_pkg is
    -- Constants for molecular biology
    constant AMINO_ACID_BITS    : integer := 5;
    constant NUCLEOTIDE_BITS    : integer := 3;
    constant MAX_SEQUENCE_LEN   : integer := 10000;
    constant DATA_WIDTH         : integer := 32;
    constant ADDR_WIDTH         : integer := 16;
    constant FIFO_DEPTH         : integer := 1024;
    
    -- Amino acid encoding
    constant AA_ALA : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "00000";
    constant AA_ARG : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "00001";
    constant AA_ASN : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "00010";
    constant AA_ASP : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "00011";
    constant AA_CYS : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "00100";
    constant AA_GLN : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "00101";
    constant AA_GLU : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "00110";
    constant AA_GLY : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "00111";
    constant AA_HIS : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "01000";
    constant AA_ILE : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "01001";
    constant AA_LEU : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "01010";
    constant AA_LYS : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "01011";
    constant AA_MET : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "01100";
    constant AA_PHE : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "01101";
    constant AA_PRO : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "01110";
    constant AA_SER : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "01111";
    constant AA_THR : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "10000";
    constant AA_TRP : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "10001";
    constant AA_TYR : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "10010";
    constant AA_VAL : std_logic_vector(AMINO_ACID_BITS-1 downto 0) := "10011";
    
    -- Nucleotide encoding
    constant NT_A : std_logic_vector(NUCLEOTIDE_BITS-1 downto 0) := "000";
    constant NT_C : std_logic_vector(NUCLEOTIDE_BITS-1 downto 0) := "001";
    constant NT_G : std_logic_vector(NUCLEOTIDE_BITS-1 downto 0) := "010";
    constant NT_T : std_logic_vector(NUCLEOTIDE_BITS-1 downto 0) := "011";
    constant NT_U : std_logic_vector(NUCLEOTIDE_BITS-1 downto 0) := "100";
    
    -- Processing states
    type processing_state_type is (
        IDLE,
        SEQUENCE_INPUT,
        SEQUENCE_VALIDATION,
        STRUCTURE_PREDICTION,
        CONFIDENCE_CALCULATION,
        RESULT_OUTPUT,
        ERROR_STATE
    );
end package bioinformatics_pkg;

-- Main bioinformatics accelerator entity
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.bioinformatics_pkg.all;

entity jaded_bioinformatics_accelerator is
    port (
        -- Clock and reset
        clk                 : in  std_logic;
        rst_n               : in  std_logic;
        
        -- AXI4-Lite control interface
        s_axi_awaddr        : in  std_logic_vector(ADDR_WIDTH-1 downto 0);
        s_axi_awvalid       : in  std_logic;
        s_axi_awready       : out std_logic;
        s_axi_wdata         : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        s_axi_wstrb         : in  std_logic_vector(3 downto 0);
        s_axi_wvalid        : in  std_logic;
        s_axi_wready        : out std_logic;
        s_axi_bresp         : out std_logic_vector(1 downto 0);
        s_axi_bvalid        : out std_logic;
        s_axi_bready        : in  std_logic;
        s_axi_araddr        : in  std_logic_vector(ADDR_WIDTH-1 downto 0);
        s_axi_arvalid       : in  std_logic;
        s_axi_arready       : out std_logic;
        s_axi_rdata         : out std_logic_vector(DATA_WIDTH-1 downto 0);
        s_axi_rresp         : out std_logic_vector(1 downto 0);
        s_axi_rvalid        : out std_logic;
        s_axi_rready        : in  std_logic;
        
        -- High-speed sequence input
        sequence_data       : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        sequence_valid      : in  std_logic;
        sequence_ready      : out std_logic;
        sequence_last       : in  std_logic;
        
        -- AlphaFold prediction output
        prediction_data     : out std_logic_vector(DATA_WIDTH-1 downto 0);
        prediction_valid    : out std_logic;
        prediction_ready    : in  std_logic;
        prediction_last     : out std_logic;
        
        -- Status and debugging
        status              : out std_logic_vector(7 downto 0);
        processing_active   : out std_logic;
        error_flag          : out std_logic;
        debug_counter       : out std_logic_vector(15 downto 0)
    );
end entity jaded_bioinformatics_accelerator;

architecture behavioral of jaded_bioinformatics_accelerator is
    -- Internal signals
    signal control_reg          : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal status_reg           : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal sequence_length_reg  : std_logic_vector(15 downto 0);
    signal confidence_reg       : std_logic_vector(7 downto 0);
    signal start_processing     : std_logic;
    signal reset_pipeline       : std_logic;
    
    -- FIFO signals
    signal sequence_fifo_empty  : std_logic;
    signal sequence_fifo_full   : std_logic;
    signal sequence_fifo_data   : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal sequence_fifo_rd_en  : std_logic;
    signal sequence_fifo_wr_en  : std_logic;
    
    signal prediction_fifo_empty : std_logic;
    signal prediction_fifo_full  : std_logic;
    signal prediction_fifo_data  : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal prediction_fifo_rd_en : std_logic;
    signal prediction_fifo_wr_en : std_logic;
    
    -- Processing state machine
    signal current_state        : processing_state_type;
    signal next_state          : processing_state_type;
    
    -- Validation signals
    signal amino_acid_decoded   : std_logic_vector(AMINO_ACID_BITS-1 downto 0);
    signal amino_acid_valid     : std_logic;
    signal sequence_validation_complete : std_logic;
    signal sequence_is_valid    : std_logic;
    
    -- Prediction engine signals
    signal structure_coords_x   : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal structure_coords_y   : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal structure_coords_z   : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal structure_prediction_valid : std_logic;
    signal structure_prediction_complete : std_logic;
    signal prediction_confidence : std_logic_vector(7 downto 0);
    
    -- Confidence calculation signals
    signal final_confidence     : std_logic_vector(7 downto 0);
    signal confidence_calculation_complete : std_logic;
    
    -- Debug counter
    signal debug_counter_reg    : std_logic_vector(15 downto 0);

    -- Component declarations
    component sync_fifo is
        generic (
            DATA_WIDTH : integer := 32;
            FIFO_DEPTH : integer := 1024
        );
        port (
            clk     : in  std_logic;
            rst_n   : in  std_logic;
            din     : in  std_logic_vector(DATA_WIDTH-1 downto 0);
            wr_en   : in  std_logic;
            rd_en   : in  std_logic;
            dout    : out std_logic_vector(DATA_WIDTH-1 downto 0);
            full    : out std_logic;
            empty   : out std_logic
        );
    end component;

    component protein_sequence_validator is
        port (
            clk                     : in  std_logic;
            rst_n                   : in  std_logic;
            enable                  : in  std_logic;
            sequence_data           : in  std_logic_vector(DATA_WIDTH-1 downto 0);
            sequence_valid          : in  std_logic;
            sequence_ready          : out std_logic;
            amino_acid_out          : out std_logic_vector(AMINO_ACID_BITS-1 downto 0);
            amino_acid_valid        : out std_logic;
            validation_complete     : out std_logic;
            sequence_is_valid       : out std_logic;
            sequence_length         : out std_logic_vector(15 downto 0)
        );
    end component;

    component alphafold_prediction_engine is
        port (
            clk                     : in  std_logic;
            rst_n                   : in  std_logic;
            enable                  : in  std_logic;
            amino_acid_in           : in  std_logic_vector(AMINO_ACID_BITS-1 downto 0);
            amino_acid_valid        : in  std_logic;
            sequence_length         : in  std_logic_vector(15 downto 0);
            structure_x             : out std_logic_vector(DATA_WIDTH-1 downto 0);
            structure_y             : out std_logic_vector(DATA_WIDTH-1 downto 0);
            structure_z             : out std_logic_vector(DATA_WIDTH-1 downto 0);
            structure_valid         : out std_logic;
            prediction_complete     : out std_logic;
            confidence              : out std_logic_vector(7 downto 0)
        );
    end component;

    component confidence_calculator is
        port (
            clk                     : in  std_logic;
            rst_n                   : in  std_logic;
            enable                  : in  std_logic;
            structure_coords_x      : in  std_logic_vector(DATA_WIDTH-1 downto 0);
            structure_coords_y      : in  std_logic_vector(DATA_WIDTH-1 downto 0);
            structure_coords_z      : in  std_logic_vector(DATA_WIDTH-1 downto 0);
            structure_valid         : in  std_logic;
            sequence_length         : in  std_logic_vector(15 downto 0);
            raw_confidence          : in  std_logic_vector(7 downto 0);
            final_confidence        : out std_logic_vector(7 downto 0);
            calculation_complete    : out std_logic
        );
    end component;

    component axi4_lite_slave is
        port (
            clk             : in  std_logic;
            rst_n           : in  std_logic;
            s_axi_awaddr    : in  std_logic_vector(ADDR_WIDTH-1 downto 0);
            s_axi_awvalid   : in  std_logic;
            s_axi_awready   : out std_logic;
            s_axi_wdata     : in  std_logic_vector(DATA_WIDTH-1 downto 0);
            s_axi_wstrb     : in  std_logic_vector(3 downto 0);
            s_axi_wvalid    : in  std_logic;
            s_axi_wready    : out std_logic;
            s_axi_bresp     : out std_logic_vector(1 downto 0);
            s_axi_bvalid    : out std_logic;
            s_axi_bready    : in  std_logic;
            s_axi_araddr    : in  std_logic_vector(ADDR_WIDTH-1 downto 0);
            s_axi_arvalid   : in  std_logic;
            s_axi_arready   : out std_logic;
            s_axi_rdata     : out std_logic_vector(DATA_WIDTH-1 downto 0);
            s_axi_rresp     : out std_logic_vector(1 downto 0);
            s_axi_rvalid    : out std_logic;
            s_axi_rready    : in  std_logic;
            control_reg     : out std_logic_vector(DATA_WIDTH-1 downto 0);
            status_reg      : in  std_logic_vector(DATA_WIDTH-1 downto 0);
            start_processing : out std_logic;
            reset_pipeline  : out std_logic
        );
    end component;

begin
    -- Component instantiations
    
    -- AXI4-Lite control interface
    axi_ctrl_interface : axi4_lite_slave
        port map (
            clk             => clk,
            rst_n           => rst_n,
            s_axi_awaddr    => s_axi_awaddr,
            s_axi_awvalid   => s_axi_awvalid,
            s_axi_awready   => s_axi_awready,
            s_axi_wdata     => s_axi_wdata,
            s_axi_wstrb     => s_axi_wstrb,
            s_axi_wvalid    => s_axi_wvalid,
            s_axi_wready    => s_axi_wready,
            s_axi_bresp     => s_axi_bresp,
            s_axi_bvalid    => s_axi_bvalid,
            s_axi_bready    => s_axi_bready,
            s_axi_araddr    => s_axi_araddr,
            s_axi_arvalid   => s_axi_arvalid,
            s_axi_arready   => s_axi_arready,
            s_axi_rdata     => s_axi_rdata,
            s_axi_rresp     => s_axi_rresp,
            s_axi_rvalid    => s_axi_rvalid,
            s_axi_rready    => s_axi_rready,
            control_reg     => control_reg,
            status_reg      => status_reg,
            start_processing => start_processing,
            reset_pipeline  => reset_pipeline
        );

    -- High-speed sequence input FIFO
    sequence_input_fifo : sync_fifo
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            FIFO_DEPTH => FIFO_DEPTH
        )
        port map (
            clk     => clk,
            rst_n   => rst_n and not reset_pipeline,
            din     => sequence_data,
            wr_en   => sequence_fifo_wr_en,
            rd_en   => sequence_fifo_rd_en,
            dout    => sequence_fifo_data,
            full    => sequence_fifo_full,
            empty   => sequence_fifo_empty
        );

    sequence_fifo_wr_en <= sequence_valid and sequence_ready;
    sequence_ready <= not sequence_fifo_full;

    -- Protein sequence validation engine
    seq_validator : protein_sequence_validator
        port map (
            clk                     => clk,
            rst_n                   => rst_n,
            enable                  => '1' when current_state = SEQUENCE_VALIDATION else '0',
            sequence_data           => sequence_fifo_data,
            sequence_valid          => not sequence_fifo_empty,
            sequence_ready          => sequence_fifo_rd_en,
            amino_acid_out          => amino_acid_decoded,
            amino_acid_valid        => amino_acid_valid,
            validation_complete     => sequence_validation_complete,
            sequence_is_valid       => sequence_is_valid,
            sequence_length         => sequence_length_reg
        );

    -- AlphaFold 3++ structure prediction engine
    alphafold_engine : alphafold_prediction_engine
        port map (
            clk                     => clk,
            rst_n                   => rst_n,
            enable                  => '1' when current_state = STRUCTURE_PREDICTION else '0',
            amino_acid_in           => amino_acid_decoded,
            amino_acid_valid        => amino_acid_valid,
            sequence_length         => sequence_length_reg,
            structure_x             => structure_coords_x,
            structure_y             => structure_coords_y,
            structure_z             => structure_coords_z,
            structure_valid         => structure_prediction_valid,
            prediction_complete     => structure_prediction_complete,
            confidence              => prediction_confidence
        );

    -- Confidence calculation and quality assessment
    conf_calc : confidence_calculator
        port map (
            clk                     => clk,
            rst_n                   => rst_n,
            enable                  => '1' when current_state = CONFIDENCE_CALCULATION else '0',
            structure_coords_x      => structure_coords_x,
            structure_coords_y      => structure_coords_y,
            structure_coords_z      => structure_coords_z,
            structure_valid         => structure_prediction_valid,
            sequence_length         => sequence_length_reg,
            raw_confidence          => prediction_confidence,
            final_confidence        => final_confidence,
            calculation_complete    => confidence_calculation_complete
        );

    -- Result output FIFO
    prediction_output_fifo : sync_fifo
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            FIFO_DEPTH => FIFO_DEPTH
        )
        port map (
            clk     => clk,
            rst_n   => rst_n,
            din     => prediction_fifo_data,
            wr_en   => prediction_fifo_wr_en,
            rd_en   => prediction_fifo_rd_en,
            dout    => prediction_data,
            full    => prediction_fifo_full,
            empty   => prediction_fifo_empty
        );

    prediction_fifo_rd_en <= prediction_ready and prediction_valid;
    prediction_valid <= not prediction_fifo_empty;

    -- Main processing state machine
    state_machine_sync : process(clk, rst_n)
    begin
        if rst_n = '0' then
            current_state <= IDLE;
        elsif rising_edge(clk) then
            current_state <= next_state;
        end if;
    end process state_machine_sync;

    state_machine_comb : process(current_state, start_processing, sequence_fifo_empty, 
                                sequence_last, sequence_valid, sequence_validation_complete,
                                sequence_is_valid, structure_prediction_complete,
                                confidence_calculation_complete, prediction_last,
                                prediction_valid, prediction_ready, reset_pipeline)
    begin
        next_state <= current_state;
        
        case current_state is
            when IDLE =>
                if start_processing = '1' and sequence_fifo_empty = '0' then
                    next_state <= SEQUENCE_INPUT;
                end if;
                
            when SEQUENCE_INPUT =>
                if sequence_last = '1' and sequence_valid = '1' then
                    next_state <= SEQUENCE_VALIDATION;
                end if;
                
            when SEQUENCE_VALIDATION =>
                if sequence_validation_complete = '1' then
                    if sequence_is_valid = '1' then
                        next_state <= STRUCTURE_PREDICTION;
                    else
                        next_state <= ERROR_STATE;
                    end if;
                end if;
                
            when STRUCTURE_PREDICTION =>
                if structure_prediction_complete = '1' then
                    next_state <= CONFIDENCE_CALCULATION;
                end if;
                
            when CONFIDENCE_CALCULATION =>
                if confidence_calculation_complete = '1' then
                    next_state <= RESULT_OUTPUT;
                end if;
                
            when RESULT_OUTPUT =>
                if prediction_last = '1' and prediction_valid = '1' and prediction_ready = '1' then
                    next_state <= IDLE;
                end if;
                
            when ERROR_STATE =>
                if reset_pipeline = '1' then
                    next_state <= IDLE;
                end if;
                
        end case;
    end process state_machine_comb;

    -- Status and control register updates
    status_control_regs : process(clk, rst_n)
    begin
        if rst_n = '0' then
            status_reg <= (others => '0');
            confidence_reg <= (others => '0');
        elsif rising_edge(clk) then
            -- Update status register
            case current_state is
                when IDLE                   => status_reg(3 downto 0) <= "0000";
                when SEQUENCE_INPUT         => status_reg(3 downto 0) <= "0001";
                when SEQUENCE_VALIDATION    => status_reg(3 downto 0) <= "0010";
                when STRUCTURE_PREDICTION   => status_reg(3 downto 0) <= "0011";
                when CONFIDENCE_CALCULATION => status_reg(3 downto 0) <= "0100";
                when RESULT_OUTPUT          => status_reg(3 downto 0) <= "0101";
                when ERROR_STATE           => status_reg(3 downto 0) <= "0111";
            end case;
            
            status_reg(4) <= '1' when (current_state /= IDLE and current_state /= ERROR_STATE) else '0';
            status_reg(5) <= '1' when current_state = ERROR_STATE else '0';
            status_reg(6) <= sequence_fifo_empty;
            status_reg(7) <= sequence_fifo_full;
            status_reg(15 downto 8) <= confidence_reg;
            
            if confidence_calculation_complete = '1' then
                confidence_reg <= final_confidence;
            end if;
        end if;
    end process status_control_regs;

    -- Output assignments
    status <= status_reg(7 downto 0);
    processing_active <= '1' when (current_state /= IDLE and current_state /= ERROR_STATE) else '0';
    error_flag <= '1' when current_state = ERROR_STATE else '0';
    prediction_last <= '1' when (current_state = RESULT_OUTPUT and prediction_fifo_rd_en = '1') else '0';

    -- Debug counter for monitoring
    debug_counter_proc : process(clk, rst_n)
    begin
        if rst_n = '0' then
            debug_counter_reg <= (others => '0');
        elsif rising_edge(clk) then
            if processing_active = '1' then
                debug_counter_reg <= std_logic_vector(unsigned(debug_counter_reg) + 1);
            end if;
        end if;
    end process debug_counter_proc;
    
    debug_counter <= debug_counter_reg;

end architecture behavioral;

-- Protein sequence validation component
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.bioinformatics_pkg.all;

entity protein_sequence_validator is
    port (
        clk                     : in  std_logic;
        rst_n                   : in  std_logic;
        enable                  : in  std_logic;
        sequence_data           : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        sequence_valid          : in  std_logic;
        sequence_ready          : out std_logic;
        amino_acid_out          : out std_logic_vector(AMINO_ACID_BITS-1 downto 0);
        amino_acid_valid        : out std_logic;
        validation_complete     : out std_logic;
        sequence_is_valid       : out std_logic;
        sequence_length         : out std_logic_vector(15 downto 0)
    );
end entity protein_sequence_validator;

architecture behavioral of protein_sequence_validator is
    signal validation_active    : std_logic;
    signal current_char         : std_logic_vector(7 downto 0);
    signal char_is_valid_aa     : std_logic;
    signal length_valid         : std_logic;
    signal sequence_length_reg  : std_logic_vector(15 downto 0);
    
    -- ASCII to amino acid lookup function
    function ascii_to_aa(ascii_char : std_logic_vector(7 downto 0)) 
        return std_logic_vector is
    begin
        case ascii_char is
            when X"41" => return AA_ALA; -- 'A'
            when X"52" => return AA_ARG; -- 'R'
            when X"4E" => return AA_ASN; -- 'N'
            when X"44" => return AA_ASP; -- 'D'
            when X"43" => return AA_CYS; -- 'C'
            when X"51" => return AA_GLN; -- 'Q'
            when X"45" => return AA_GLU; -- 'E'
            when X"47" => return AA_GLY; -- 'G'
            when X"48" => return AA_HIS; -- 'H'
            when X"49" => return AA_ILE; -- 'I'
            when X"4C" => return AA_LEU; -- 'L'
            when X"4B" => return AA_LYS; -- 'K'
            when X"4D" => return AA_MET; -- 'M'
            when X"46" => return AA_PHE; -- 'F'
            when X"50" => return AA_PRO; -- 'P'
            when X"53" => return AA_SER; -- 'S'
            when X"54" => return AA_THR; -- 'T'
            when X"57" => return AA_TRP; -- 'W'
            when X"59" => return AA_TYR; -- 'Y'
            when X"56" => return AA_VAL; -- 'V'
            when others => return (others => '0');
        end case;
    end function;
    
    -- Check if character is valid amino acid
    function is_valid_aa(ascii_char : std_logic_vector(7 downto 0)) 
        return std_logic is
    begin
        case ascii_char is
            when X"41" | X"52" | X"4E" | X"44" | X"43" | X"51" | X"45" | X"47" | X"48" | X"49" |
                 X"4C" | X"4B" | X"4D" | X"46" | X"50" | X"53" | X"54" | X"57" | X"59" | X"56" =>
                return '1';
            when others => return '0';
        end case;
    end function;

begin
    validation_proc : process(clk, rst_n)
    begin
        if rst_n = '0' then
            sequence_length_reg <= (others => '0');
            validation_active <= '0';
            current_char <= (others => '0');
            char_is_valid_aa <= '0';
            length_valid <= '1';
        elsif rising_edge(clk) then
            if enable = '1' and sequence_valid = '1' then
                validation_active <= '1';
                current_char <= sequence_data(7 downto 0);
                
                -- Check if current character is a valid amino acid
                if is_valid_aa(sequence_data(7 downto 0)) = '1' then
                    char_is_valid_aa <= '1';
                    sequence_length_reg <= std_logic_vector(unsigned(sequence_length_reg) + 1);
                    
                    if unsigned(sequence_length_reg) >= MAX_SEQUENCE_LEN then
                        length_valid <= '0';
                    end if;
                else
                    char_is_valid_aa <= '0';
                end if;
            elsif enable = '0' then
                validation_active <= '0';
                sequence_length_reg <= (others => '0');
                length_valid <= '1';
            end if;
        end if;
    end process validation_proc;

    amino_acid_out <= ascii_to_aa(current_char);
    amino_acid_valid <= validation_active and char_is_valid_aa;
    sequence_ready <= enable;
    validation_complete <= not sequence_valid and validation_active;
    sequence_is_valid <= validation_complete and 
                        (unsigned(sequence_length_reg) > 0) and 
                        (unsigned(sequence_length_reg) <= MAX_SEQUENCE_LEN) and 
                        length_valid;
    sequence_length <= sequence_length_reg;

end architecture behavioral;

-- AlphaFold prediction engine (simplified for demonstration)
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.bioinformatics_pkg.all;

entity alphafold_prediction_engine is
    port (
        clk                     : in  std_logic;
        rst_n                   : in  std_logic;
        enable                  : in  std_logic;
        amino_acid_in           : in  std_logic_vector(AMINO_ACID_BITS-1 downto 0);
        amino_acid_valid        : in  std_logic;
        sequence_length         : in  std_logic_vector(15 downto 0);
        structure_x             : out std_logic_vector(DATA_WIDTH-1 downto 0);
        structure_y             : out std_logic_vector(DATA_WIDTH-1 downto 0);
        structure_z             : out std_logic_vector(DATA_WIDTH-1 downto 0);
        structure_valid         : out std_logic;
        prediction_complete     : out std_logic;
        confidence              : out std_logic_vector(7 downto 0)
    );
end entity alphafold_prediction_engine;

architecture behavioral of alphafold_prediction_engine is
    signal processing_counter   : std_logic_vector(15 downto 0);
    signal current_residue      : std_logic_vector(15 downto 0);
    signal prediction_active    : std_logic;
    signal structure_x_reg      : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal structure_y_reg      : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal structure_z_reg      : std_logic_vector(DATA_WIDTH-1 downto 0);

begin
    prediction_proc : process(clk, rst_n)
    begin
        if rst_n = '0' then
            structure_x_reg <= (others => '0');
            structure_y_reg <= (others => '0');
            structure_z_reg <= (others => '0');
            structure_valid <= '0';
            prediction_complete <= '0';
            confidence <= X"5F"; -- Default confidence of 95
            processing_counter <= (others => '0');
            current_residue <= (others => '0');
            prediction_active <= '0';
        elsif rising_edge(clk) then
            if enable = '1' then
                prediction_active <= '1';
                
                if amino_acid_valid = '1' then
                    -- Simplified coordinate calculation based on amino acid type and position
                    structure_x_reg <= std_logic_vector(unsigned(current_residue) + 
                                      unsigned(amino_acid_in & "000"));
                    structure_y_reg <= std_logic_vector(unsigned(current_residue) + 
                                      unsigned(amino_acid_in & "001"));
                    structure_z_reg <= std_logic_vector(unsigned(current_residue) + 
                                      unsigned(amino_acid_in & "010"));
                    structure_valid <= '1';
                    current_residue <= std_logic_vector(unsigned(current_residue) + 1);
                else
                    structure_valid <= '0';
                end if;
                
                processing_counter <= std_logic_vector(unsigned(processing_counter) + 1);
                
                -- Complete prediction when all residues processed
                if unsigned(current_residue) >= unsigned(sequence_length) then
                    prediction_complete <= '1';
                end if;
            else
                prediction_active <= '0';
                prediction_complete <= '0';
                processing_counter <= (others => '0');
                current_residue <= (others => '0');
                structure_valid <= '0';
            end if;
        end if;
    end process prediction_proc;

    structure_x <= structure_x_reg;
    structure_y <= structure_y_reg;
    structure_z <= structure_z_reg;

end architecture behavioral;

-- Confidence calculation component
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.bioinformatics_pkg.all;

entity confidence_calculator is
    port (
        clk                     : in  std_logic;
        rst_n                   : in  std_logic;
        enable                  : in  std_logic;
        structure_coords_x      : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        structure_coords_y      : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        structure_coords_z      : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        structure_valid         : in  std_logic;
        sequence_length         : in  std_logic_vector(15 downto 0);
        raw_confidence          : in  std_logic_vector(7 downto 0);
        final_confidence        : out std_logic_vector(7 downto 0);
        calculation_complete    : out std_logic
    );
end entity confidence_calculator;

architecture behavioral of confidence_calculator is
    signal calc_counter           : std_logic_vector(15 downto 0);
    signal confidence_accumulator : std_logic_vector(31 downto 0);
    signal final_confidence_reg   : std_logic_vector(7 downto 0);

begin
    confidence_proc : process(clk, rst_n)
    begin
        if rst_n = '0' then
            final_confidence_reg <= (others => '0');
            calculation_complete <= '0';
            calc_counter <= (others => '0');
            confidence_accumulator <= (others => '0');
        elsif rising_edge(clk) then
            if enable = '1' then
                if structure_valid = '1' then
                    -- Accumulate confidence based on structure quality metrics
                    confidence_accumulator <= std_logic_vector(unsigned(confidence_accumulator) + 
                                            unsigned(raw_confidence));
                    calc_counter <= std_logic_vector(unsigned(calc_counter) + 1);
                end if;
                
                -- Finalize confidence calculation
                if unsigned(calc_counter) >= unsigned(sequence_length) then
                    final_confidence_reg <= confidence_accumulator(15 downto 8); -- Average confidence
                    calculation_complete <= '1';
                end if;
            else
                calculation_complete <= '0';
                calc_counter <= (others => '0');
                confidence_accumulator <= (others => '0');
            end if;
        end if;
    end process confidence_proc;

    final_confidence <= final_confidence_reg;

end architecture behavioral;

-- Generic synchronous FIFO component
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity sync_fifo is
    generic (
        DATA_WIDTH : integer := 32;
        FIFO_DEPTH : integer := 1024
    );
    port (
        clk     : in  std_logic;
        rst_n   : in  std_logic;
        din     : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        wr_en   : in  std_logic;
        rd_en   : in  std_logic;
        dout    : out std_logic_vector(DATA_WIDTH-1 downto 0);
        full    : out std_logic;
        empty   : out std_logic
    );
end entity sync_fifo;

architecture behavioral of sync_fifo is
    constant ADDR_WIDTH : integer := integer(ceil(log2(real(FIFO_DEPTH))));
    
    type fifo_mem_type is array (0 to FIFO_DEPTH-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal fifo_mem : fifo_mem_type;
    
    signal wr_ptr : std_logic_vector(ADDR_WIDTH downto 0);
    signal rd_ptr : std_logic_vector(ADDR_WIDTH downto 0);
    signal full_flag : std_logic;
    signal empty_flag : std_logic;

begin
    fifo_proc : process(clk, rst_n)
    begin
        if rst_n = '0' then
            wr_ptr <= (others => '0');
            rd_ptr <= (others => '0');
            dout <= (others => '0');
        elsif rising_edge(clk) then
            if wr_en = '1' and full_flag = '0' then
                fifo_mem(to_integer(unsigned(wr_ptr(ADDR_WIDTH-1 downto 0)))) <= din;
                wr_ptr <= std_logic_vector(unsigned(wr_ptr) + 1);
            end if;
            
            if rd_en = '1' and empty_flag = '0' then
                dout <= fifo_mem(to_integer(unsigned(rd_ptr(ADDR_WIDTH-1 downto 0))));
                rd_ptr <= std_logic_vector(unsigned(rd_ptr) + 1);
            end if;
        end if;
    end process fifo_proc;

    full_flag <= '1' when (wr_ptr(ADDR_WIDTH) /= rd_ptr(ADDR_WIDTH)) and 
                         (wr_ptr(ADDR_WIDTH-1 downto 0) = rd_ptr(ADDR_WIDTH-1 downto 0)) else '0';
    empty_flag <= '1' when wr_ptr = rd_ptr else '0';
    
    full <= full_flag;
    empty <= empty_flag;

end architecture behavioral;

-- Simplified AXI4-Lite slave interface
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.bioinformatics_pkg.all;

entity axi4_lite_slave is
    port (
        clk             : in  std_logic;
        rst_n           : in  std_logic;
        s_axi_awaddr    : in  std_logic_vector(ADDR_WIDTH-1 downto 0);
        s_axi_awvalid   : in  std_logic;
        s_axi_awready   : out std_logic;
        s_axi_wdata     : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        s_axi_wstrb     : in  std_logic_vector(3 downto 0);
        s_axi_wvalid    : in  std_logic;
        s_axi_wready    : out std_logic;
        s_axi_bresp     : out std_logic_vector(1 downto 0);
        s_axi_bvalid    : out std_logic;
        s_axi_bready    : in  std_logic;
        s_axi_araddr    : in  std_logic_vector(ADDR_WIDTH-1 downto 0);
        s_axi_arvalid   : in  std_logic;
        s_axi_arready   : out std_logic;
        s_axi_rdata     : out std_logic_vector(DATA_WIDTH-1 downto 0);
        s_axi_rresp     : out std_logic_vector(1 downto 0);
        s_axi_rvalid    : out std_logic;
        s_axi_rready    : in  std_logic;
        control_reg     : out std_logic_vector(DATA_WIDTH-1 downto 0);
        status_reg      : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        start_processing : out std_logic;
        reset_pipeline  : out std_logic
    );
end entity axi4_lite_slave;

architecture behavioral of axi4_lite_slave is
    signal control_reg_int      : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal start_processing_int : std_logic;
    signal reset_pipeline_int   : std_logic;

begin
    axi_proc : process(clk, rst_n)
    begin
        if rst_n = '0' then
            s_axi_awready <= '0';
            s_axi_wready <= '0';
            s_axi_bvalid <= '0';
            s_axi_bresp <= "00";
            s_axi_arready <= '0';
            s_axi_rvalid <= '0';
            s_axi_rdata <= (others => '0');
            s_axi_rresp <= "00";
            control_reg_int <= (others => '0');
            start_processing_int <= '0';
            reset_pipeline_int <= '0';
        elsif rising_edge(clk) then
            -- Write channel
            if s_axi_awvalid = '1' and s_axi_wvalid = '1' and s_axi_bvalid = '0' then
                s_axi_awready <= '1';
                s_axi_wready <= '1';
                s_axi_bvalid <= '1';
                
                -- Handle control register writes
                if s_axi_awaddr = X"0000" then
                    control_reg_int <= s_axi_wdata;
                    start_processing_int <= s_axi_wdata(0);
                    reset_pipeline_int <= s_axi_wdata(1);
                end if;
            elsif s_axi_bready = '1' and s_axi_bvalid = '1' then
                s_axi_awready <= '0';
                s_axi_wready <= '0';
                s_axi_bvalid <= '0';
            end if;
            
            -- Read channel
            if s_axi_arvalid = '1' and s_axi_rvalid = '0' then
                s_axi_arready <= '1';
                s_axi_rvalid <= '1';
                
                -- Handle status register reads
                if s_axi_araddr = X"0004" then
                    s_axi_rdata <= status_reg;
                elsif s_axi_araddr = X"0000" then
                    s_axi_rdata <= control_reg_int;
                end if;
            elsif s_axi_rready = '1' and s_axi_rvalid = '1' then
                s_axi_arready <= '0';
                s_axi_rvalid <= '0';
            end if;
        end if;
    end process axi_proc;

    control_reg <= control_reg_int;
    start_processing <= start_processing_int;
    reset_pipeline <= reset_pipeline_int;

end architecture behavioral;