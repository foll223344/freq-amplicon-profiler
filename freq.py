import argparse
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
import os
import numpy as np
import re
from Bio.Seq import reverse_complement
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

IUPAC = {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
    'R': '[AG]', 'Y': '[CT]', 'S': '[GC]', 'W': '[AT]',
    'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]',
    'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]'
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze genetic variants and motifs in a genomic region.')
    parser.add_argument('--input', required=True, help='Input TSV file path')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--region', required=True, 
                      help='Genomic region in format chr:start-end (e.g., chr4:100767300-100767370)')
    parser.add_argument('--ref-genome', required=True, help='Reference genome FASTA file')
    parser.add_argument('--motif1', default='', help='First motif to search (IUPAC notation)')
    parser.add_argument('--motif2', default='', help='Second motif to search')
    parser.add_argument('--alt-reads-mode', action='store_true',
                      help='Calculate alt frequency as sum of all alternative reads')
    parser.add_argument('--dpi', type=int, default=300, help='Output image resolution')
    parser.add_argument('--noise-threshold', type=float, default=0.0, 
                      help='Threshold for mutations per thousand (values below this will be ignored)')
    parser.add_argument('--log-scale', action='store_true', 
                      help='Use logarithmic scale for Y-axis')
    parser.add_argument('--disable-axis-labels', action='store_true', 
                      help='Disable axis labels on the plot and save them to a separate TSV file')
    parser.add_argument('--hide-ref-nt', action='store_true',
                      help='Hide reference nucleotides on the plot')
    parser.add_argument('--plot-type', choices=['bar', 'area'], default='bar',
                      help='Plot type: bar or area (smoothed line with filled area)')
    parser.add_argument('--multi-sample', action='store_true',
                      help='Plot multiple samples on same graph')
    parser.add_argument('--sample-names', default='',
                      help='Comma-separated sample names to include in multi-sample plot')
    parser.add_argument('--colors', default='red,blue,green,orange',
                      help='Comma-separated colors for multi-sample plot')
    parser.add_argument('--smooth-sigma', type=float, default=2.0,
                      help='Sigma for Gaussian smoothing (for area plots)')
    parser.add_argument('--svg-format', action='store_true',
                      help='Also save plots in SVG format')
    parser.add_argument('--figure-width', type=float, default=None,
                      help='Figure width in inches (default: auto-calculated based on region length)')
    parser.add_argument('--figure-height', type=float, default=4,
                      help='Figure height in inches (default: 4)')
    parser.add_argument('--solid-colors', action='store_true',
                      help='Use solid colors without transparency')
    parser.add_argument('--minimal-mode', action='store_true',
                      help='Minimal mode: no title, no legend, simplified axis')
    parser.add_argument('--y-max', '--y_max', dest='y_max', type=float, default=None,
                      help='Hard upper limit for the Y axis (mutations per thousand), '
                           'e.g. --y_max 100. If not set, the limit is auto-scaled.')
    parser.add_argument('--y-min', '--y_min', dest='y_min', type=float, default=None,
                      help='Lower limit for the Y axis (default: 0, or 0.1 with --log-scale)')
    return parser.parse_args()

def parse_region(region_str):
    match = re.match(r'([^:]+):(\d+)-(\d+)', region_str)
    if not match:
        raise ValueError("Invalid region format. Use chr:start-end")
    return {
        'chrom': match.group(1),
        'start': int(match.group(2)),
        'end': int(match.group(3))
    }

def motif_to_regex(motif):
    return ''.join(IUPAC.get(char.upper(), char) for char in motif)

def find_motifs(sequence, motif, region):
    if not motif:
        return []
    regex = motif_to_regex(motif)
    matches = []
    for match in re.finditer(regex, sequence):
        start = match.start() + region['start']
        end = match.end() + region['start'] - 1
        matches.append(("forward", start, end))
    rev_sequence = reverse_complement(sequence)
    for match in re.finditer(regex, rev_sequence):
        rev_start = len(sequence) - match.end()
        rev_end = len(sequence) - match.start() - 1
        matches.append(("reverse", region['start'] + rev_start, region['start'] + rev_end))
    return matches

def load_reference(fasta_file, chrom):
    for record in SeqIO.parse(fasta_file, "fasta"):
        if record.id == chrom:
            return str(record.seq)
    raise ValueError(f"Chromosome {chrom} not found in FASTA!")

def calculate_alt_metrics(ad_str, alt_reads_mode):
    try:
        ad = list(map(int, ad_str.split(',')))
        if len(ad) < 2 or sum(ad) == 0:
            return 0.0, 0
        total = sum(ad)
        if alt_reads_mode:
            alt_sum = sum(ad[1:])  # Sum all alternative alleles
            return (alt_sum / total) * 100, total
        else:
            return (ad[1] / total) * 100, total  # Original behavior (first alt only)
    except:
        return 0.0, 0

def get_indel_positions(row, ref_genome, region):
    pos = row["POS"]
    ref = row["REF"]
    alt = row["ALT"]
    min_len = min(len(ref), len(alt))
    common_prefix = 0
    while common_prefix < min_len and ref[common_prefix] == alt[common_prefix]:
        common_prefix += 1
    common_suffix = 0
    while common_suffix < min_len - common_prefix and ref[-common_suffix-1] == alt[-common_suffix-1]:
        common_suffix += 1
    if len(ref) > len(alt):
        del_len = len(ref) - len(alt)
        start = pos + common_prefix
        end = start + del_len - 1
        return list(range(start, end + 1))
    elif len(alt) > len(ref):
        ins_pos = pos + common_prefix - 1
        return [ins_pos]
    else:
        return [pos + i for i in range(len(ref))]

def process_sample_data(sample_df, region, args, ref_genome_full):
    """Process data for a single sample and return bar data"""
    positions = np.arange(region['start'], region['end'] + 1)
    position_data = defaultdict(list)
    
    for _, row in sample_df.iterrows():
        if len(row["REF"]) != len(row["ALT"]):
            affected = get_indel_positions(row, ref_genome_full, region)
        else:
            affected = [row["POS"] + i for i in range(len(row["REF"]))]
        
        for p in affected:
            if region['start'] <= p <= region['end']:
                mutations_per_thousand = (row["ALT_DP"] / row["TOTAL_DP"]) * 1000
                if mutations_per_thousand >= args.noise_threshold:
                    position_data[p].append({
                        "mutations_per_thousand": mutations_per_thousand,
                        "alt_dp": row["ALT_DP"],
                        "is_indel": len(row["REF"]) != len(row["ALT"])
                    })
    
    # Combine data for each position
    processed_data = {}
    for pos in positions:
        entries = position_data.get(pos, [])
        if entries:
            total_mutations_per_thousand = sum(e["mutations_per_thousand"] for e in entries)
            total_alt_dp = sum(e["alt_dp"] for e in entries)
            processed_data[pos] = {
                "mutations_per_thousand": total_mutations_per_thousand,
                "alt_dp": total_alt_dp,
                "label": f"{total_mutations_per_thousand:.1f} ({total_alt_dp})",
                "is_indel": any(e["is_indel"] for e in entries)
            }
        else:
            processed_data[pos] = {
                "mutations_per_thousand": 0,
                "alt_dp": 0,
                "label": "0 (0)",
                "is_indel": False
            }
    
    return processed_data

def draw_ref_nucleotides(ax, positions, ref_sequence, args, default_y):
    """Draw reference nucleotides under the plot.

    When --y_max is set, the row is anchored to the axes (just below the X axis)
    so it does not depend on the data scale; otherwise the historical fixed
    data-coordinate offset is kept.
    """
    if getattr(args, 'y_max', None) is not None:
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for pos, nt in zip(positions, ref_sequence):
            ax.text(pos, -0.28, nt, ha='center', va='center',
                    fontsize=6, color='black', transform=trans, clip_on=False)
    else:
        for pos, nt in zip(positions, ref_sequence):
            ax.text(pos, default_y, nt, ha='center', va='center',
                    fontsize=6, color='black')

def create_single_sample_bar_plot(ax, sample, sample_data, region, args, motif1_regions, motif2_regions, ref_sequence):
    """Create bar plot for single sample"""
    positions = np.arange(region['start'], region['end'] + 1)
    
    # Set alpha value based on args
    alpha = 1.0 if args.solid_colors else 0.5
    
    # Plot motifs
    for start, end in motif1_regions:
        ax.axvspan(max(start, region['start']) - 0.5, 
                 min(end, region['end']) + 0.5,
                 facecolor="yellow", alpha=0.3, label='Motif 1' if start == motif1_regions[0][0] else "")
    
    for start, end in motif2_regions:
        ax.axvspan(max(start, region['start']) - 0.5,
                 min(end, region['end']) + 0.5,
                 facecolor="blue", alpha=0.3, label='Motif 2' if start == motif2_regions[0][0] else "")
    
    # Plot reference nucleotides if not hidden
    if not args.hide_ref_nt:
        draw_ref_nucleotides(ax, positions, ref_sequence, args, default_y=-30)

    # Prepare bar data
    bar_heights = [sample_data[pos]["mutations_per_thousand"] for pos in positions]
    bar_labels = [sample_data[pos]["label"] for pos in positions]

    # Create bars
    bars = ax.bar(positions, bar_heights, width=0.8, color='red', alpha=alpha, label=sample)

    # Add labels if not disabled
    if not args.disable_axis_labels:
        y_max = getattr(args, 'y_max', None)
        for pos, bar, label in zip(positions, bars, bar_labels):
            height = sample_data[pos]["mutations_per_thousand"]
            if height > 0:
                y_pos = height / 2 if height >= 10 else height + 0.5
                va = "center" if height >= 10 else "bottom"
                # Keep labels inside the plot when a hard Y limit is set
                if y_max is not None and (height > y_max or y_pos > y_max):
                    y_pos = y_max / 2
                    va = "center"
                ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                       ha='center', va=va, fontsize=8, color='black',
                       rotation=90, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    return bars

def create_multi_sample_bar_plot(ax, all_sample_data, region, args, motif1_regions, motif2_regions, ref_sequence):
    """Create bar plot for multiple samples"""
    positions = np.arange(region['start'], region['end'] + 1)
    
    # Set alpha value based on args
    alpha = 1.0 if args.solid_colors else 0.7
    
    # Plot motifs
    for start, end in motif1_regions:
        ax.axvspan(max(start, region['start']) - 0.5, 
                 min(end, region['end']) + 0.5,
                 facecolor="yellow", alpha=0.3, label='Motif 1' if start == motif1_regions[0][0] else "")
    
    for start, end in motif2_regions:
        ax.axvspan(max(start, region['start']) - 0.5,
                 min(end, region['end']) + 0.5,
                 facecolor="blue", alpha=0.3, label='Motif 2' if start == motif2_regions[0][0] else "")
    
    # Plot reference nucleotides if not hidden
    if not args.hide_ref_nt:
        draw_ref_nucleotides(ax, positions, ref_sequence, args, default_y=-30)

    # Prepare colors
    colors = args.colors.split(',')
    sample_names = list(all_sample_data.keys())

    # Create grouped bars
    bar_width = 0.8 / len(sample_names)
    for i, (sample_name, sample_data) in enumerate(all_sample_data.items()):
        color = colors[i % len(colors)]
        bar_positions = positions + (i - len(sample_names)/2 + 0.5) * bar_width
        bar_heights = [sample_data[pos]["mutations_per_thousand"] for pos in positions]
        
        bars = ax.bar(bar_positions, bar_heights, width=bar_width,
                     color=color, alpha=alpha, label=sample_name)
    
    return sample_names

def create_area_plot(ax, all_sample_data, region, args, motif1_regions, motif2_regions, ref_sequence):
    """Create area plot (smoothed line with filled area) for multiple samples"""
    positions = np.arange(region['start'], region['end'] + 1)
    
    # Set alpha value based on args
    area_alpha = 1.0 if args.solid_colors else 0.3
    
    # Plot motifs
    for start, end in motif1_regions:
        ax.axvspan(max(start, region['start']) - 0.5, 
                 min(end, region['end']) + 0.5,
                 facecolor="yellow", alpha=0.3, label='Motif 1' if start == motif1_regions[0][0] else "")
    
    for start, end in motif2_regions:
        ax.axvspan(max(start, region['start']) - 0.5,
                 min(end, region['end']) + 0.5,
                 facecolor="blue", alpha=0.3, label='Motif 2' if start == motif2_regions[0][0] else "")
    
    # Plot reference nucleotides if not hidden
    if not args.hide_ref_nt:
        draw_ref_nucleotides(ax, positions, ref_sequence, args, default_y=-5)
    
    # Prepare colors
    colors = args.colors.split(',')
    sample_names = list(all_sample_data.keys())
    
    # Create area plots
    for i, (sample_name, sample_data) in enumerate(all_sample_data.items()):
        color = colors[i % len(colors)]
        y_values = [sample_data[pos]["mutations_per_thousand"] for pos in positions]
        
        # Apply Gaussian smoothing
        if args.smooth_sigma > 0:
            y_smooth = gaussian_filter1d(y_values, sigma=args.smooth_sigma)
        else:
            y_smooth = y_values
        
        # Plot filled area
        ax.fill_between(positions, 0, y_smooth, color=color, alpha=area_alpha)
        # Plot line
        ax.plot(positions, y_smooth, color=color, linewidth=2, label=sample_name)
    
    return sample_names

def setup_plot(ax, region, args, plot_type, sample_names):
    """Setup plot axes, labels, and title"""
    positions = np.arange(region['start'], region['end'] + 1)
    
    # Set x-axis limits
    ax.set_xlim(region['start'] - 0.5, region['end'] + 0.5)
    
    # Set x-ticks
    if args.hide_ref_nt:
        # Show only first and last positions
        if len(positions) > 1:
            tick_positions = [positions[0], positions[-1]]
            tick_labels = [str(positions[0]), str(positions[-1])]
        else:
            tick_positions = [positions[0]]
            tick_labels = [str(positions[0])]
        
        # Add intermediate ticks for very long regions
        if len(positions) > 20:
            # Add a middle tick
            mid_idx = len(positions) // 2
            tick_positions.append(positions[mid_idx])
            tick_labels.append(str(positions[mid_idx]))
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
        
        # Set axis as simple line
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
    else:
        # Show every Nth position for readability
        step = max(1, len(positions) // 20)
        visible_positions = positions[::step]
        ax.set_xticks(visible_positions)
        ax.set_xticklabels(visible_positions if not args.disable_axis_labels else [], 
                          fontsize=8, rotation=45 if len(visible_positions) > 10 else 0)
    
    # Set labels
    if not args.minimal_mode:
        ax.set_xlabel("Genomic Position", fontsize=12, labelpad=15)
        ax.set_ylabel("Mutations per Thousand Reads", fontsize=12)
    
    # Set title
    if not args.minimal_mode:
        if len(sample_names) == 1:
            title = f"Sample: {sample_names[0]}\nRegion: {region['chrom']}:{region['start']}-{region['end']}"
        else:
            title = f"Multi-sample {plot_type} plot\nRegion: {region['chrom']}:{region['start']}-{region['end']}"
        
        # Adjust title font size based on figure height
        fig = ax.get_figure()
        fig_height = fig.get_size_inches()[1]
        title_fontsize = max(8, min(12, fig_height * 2))
        ax.set_title(title, fontsize=title_fontsize, pad=10)
    
    # Set y-scale
    y_max = getattr(args, 'y_max', None)
    y_min = getattr(args, 'y_min', None)

    if args.log_scale:
        ax.set_yscale('log')
        bottom = y_min if y_min is not None and y_min > 0 else 0.1
        if y_max is not None:
            ax.set_ylim(bottom, y_max)
        else:
            # Find max non-zero value
            max_val = 0
            for line in ax.lines:
                if len(line.get_ydata()) > 0:
                    max_val = max(max_val, max(line.get_ydata()))
            for collection in ax.collections:
                if hasattr(collection, 'get_offsets'):
                    if len(collection.get_offsets()) > 0:
                        max_val = max(max_val, max(collection.get_offsets()[:, 1]))
            ax.set_ylim(bottom, max(100, max_val * 1.5) if max_val > 0 else 100)
    else:
        bottom = y_min if y_min is not None else 0
        if y_max is not None:
            ax.set_ylim(bottom, y_max)
        else:
            ax.set_ylim(bottom=bottom)
    
    # Add legend
    if (plot_type != 'single' and plot_type != 'single_area') or not args.disable_axis_labels:
        if not args.minimal_mode:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                # Remove duplicates
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)

def save_plot(fig, output_dir, sample_name, region, plot_type, svg_format=False, multi_sample_names=None):
    """Save plot in PNG and optionally SVG format"""
    if plot_type == 'single' or plot_type == 'single_area':
        base_filename = f"{sample_name}_plot"
    else:
        # Для мульти-семплов создаем имя файла с перечислением всех образцов
        if multi_sample_names and len(multi_sample_names) > 0:
            # Если много образцов, используем сокращенное имя
            if len(multi_sample_names) > 3:
                sample_part = f"{len(multi_sample_names)}_samples"
            else:
                # Для 1-3 образцов перечисляем их имена
                sample_part = "_".join(multi_sample_names)
            base_filename = f"{sample_part}_{plot_type}_plot"
        else:
            base_filename = f"multi_sample_{plot_type}_plot"
    
    # Сохраняем PNG
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    
    # Сохраняем SVG если запрошено
    if svg_format:
        svg_path = os.path.join(output_dir, f"{base_filename}.svg")
        fig.savefig(svg_path, bbox_inches='tight', format='svg')
        print(f"  Saved: {svg_path}")
    
    print(f"  Saved: {png_path}")
    plt.close(fig)

def get_figure_size(args, region_length, multi_sample=False):
    """Get figure size based on arguments or use calculated defaults"""
    if args.figure_width is not None and args.figure_height is not None:
        return (args.figure_width, args.figure_height)
    elif args.figure_width is not None:
        # Only width specified, use default height
        return (args.figure_width, args.figure_height)
    else:
        # Auto-calculate width based on region length and plot type
        if args.plot_type == 'area':
            # Area plots can be very narrow
            width_per_position = 0.02  # 0.02 inches per position
        else:
            # Bar plots need more width
            width_per_position = 0.05  # 0.05 inches per position
        
        # Calculate base width
        base_width = max(2.0, region_length * width_per_position)  # Minimum 2 inches
        
        # Adjust for multi-sample
        if multi_sample:
            base_width = min(15.0, base_width * 1.5)  # Cap at 15 inches for multi-sample
        else:
            base_width = min(12.5, base_width)  # Cap at 12.5 inches for single
        
        # Ensure reasonable minimum width
        min_width = 1.0 if args.plot_type == 'area' else 3.0
        base_width = max(min_width, base_width)
        
        # Use user-specified height or default
        height = args.figure_height if args.figure_height is not None else 4.0
        
        return (base_width, height)

def adjust_subplots_for_narrow_figure(fig, width, args):
    """Adjust subplot parameters for narrow figures"""
    if width < 4:
        # Very narrow figure
        fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.25)
    elif width < 6:
        # Narrow figure
        fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.2)
    elif width < 8:
        # Medium narrow figure
        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    else:
        # Normal figure
        if args.multi_sample:
            fig.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.15)
        else:
            fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)

def process_and_plot_samples(new_df, region, args, motif1_regions, motif2_regions, ref_genome_full, ref_sequence):
    """Process samples and create plots"""
    all_sample_data = {}
    
    # Получаем имена образцов для обработки
    if args.sample_names:
        sample_names_to_process = [name.strip() for name in args.sample_names.split(',')]
        available_samples = new_df["SAMPLE"].unique()
        sample_names = [name for name in sample_names_to_process if name in available_samples]
        if not sample_names:
            print(f"Warning: No specified samples found in data. Available: {available_samples}")
            return {}
    else:
        sample_names = sorted(new_df["SAMPLE"].unique())
    
    # Обрабатываем каждый образец
    for sample in sample_names:
        sample_df = new_df[new_df["SAMPLE"] == sample]
        if not sample_df.empty:
            all_sample_data[sample] = process_sample_data(sample_df, region, args, ref_genome_full)
        else:
            print(f"Warning: No data for sample {sample}")
    
    if not all_sample_data:
        print("No data found for any sample!")
        return {}
    
    # Расчет длины региона для авто-размера
    region_length = region['end'] - region['start'] + 1
    
    # Создаем графики в зависимости от режима
    if args.multi_sample:
        # Создаем мульти-семпл график
        fig_size = get_figure_size(args, region_length, multi_sample=True)
        fig, ax = plt.subplots(figsize=fig_size, dpi=args.dpi)
        
        # Настраиваем subplots для узких графиков
        adjust_subplots_for_narrow_figure(fig, fig_size[0], args)
        
        if args.plot_type == 'bar':
            plot_sample_names = create_multi_sample_bar_plot(ax, all_sample_data, region, args, 
                                                           motif1_regions, motif2_regions, ref_sequence)
            plot_type = 'multi_bar'
        else:  # area plot
            plot_sample_names = create_area_plot(ax, all_sample_data, region, args, 
                                               motif1_regions, motif2_regions, ref_sequence)
            plot_type = 'multi_area'
        
        setup_plot(ax, region, args, plot_type, plot_sample_names)
        save_plot(fig, args.output_dir, "", region, plot_type, args.svg_format, 
                 multi_sample_names=plot_sample_names)
        
    else:
        # Создаем отдельные графики для каждого образца
        for sample, sample_data in all_sample_data.items():
            fig_size = get_figure_size(args, region_length, multi_sample=False)
            fig, ax = plt.subplots(figsize=fig_size, dpi=args.dpi)
            
            # Настраиваем subplots для узких графиков
            adjust_subplots_for_narrow_figure(fig, fig_size[0], args)
            
            if args.plot_type == 'bar':
                create_single_sample_bar_plot(ax, sample, sample_data, region, args, 
                                            motif1_regions, motif2_regions, ref_sequence)
                plot_type = 'single'
            else:  # area plot для одного образца
                temp_data = {sample: sample_data}
                create_area_plot(ax, temp_data, region, args, 
                               motif1_regions, motif2_regions, ref_sequence)
                plot_type = 'single_area'
            
            setup_plot(ax, region, args, plot_type, [sample])
            save_plot(fig, args.output_dir, sample, region, plot_type, args.svg_format)
    
    return all_sample_data

def calculate_average_mutation_rate(sample_df, region, noise_threshold, ref_genome_full):
    """Calculate the average mutation rate for the entire region."""
    total_length = region['end'] - region['start'] + 1
    position_data = {}
    
    # Collect read data for each position
    for _, row in sample_df.iterrows():
        mutations_per_thousand = (row["ALT_DP"] / row["TOTAL_DP"]) * 1000
        if mutations_per_thousand >= noise_threshold:
            # Get affected positions for this variant
            if len(row["REF"]) != len(row["ALT"]):
                affected = get_indel_positions(row, ref_genome_full, region)
            else:
                affected = [row["POS"] + i for i in range(len(row["REF"]))]
            
            for pos in affected:
                if region['start'] <= pos <= region['end']:
                    if pos not in position_data:
                        position_data[pos] = {
                            'total_dp': row["TOTAL_DP"],
                            'alt_dp': row["ALT_DP"]
                        }
                    else:
                        # Sum the alt_dp
                        position_data[pos]['alt_dp'] += row["ALT_DP"]
    
    mutation_rates = []
    for pos in range(region['start'], region['end'] + 1):
        if pos in position_data:
            data = position_data[pos]
            mutations_per_thousand = (data['alt_dp'] / data['total_dp']) * 1000
        else:
            mutations_per_thousand = 0
        mutation_rates.append(mutations_per_thousand)
    
    # Calculate average
    average_rate = sum(mutation_rates) / total_length if total_length > 0 else 0
    return average_rate

def save_summary_tsv(summary_data, output_file):
    """Save summary data to a TSV file."""
    summary_df = pd.DataFrame(summary_data, columns=["Sample", "Average_Mutation_Rate"])
    summary_df.to_csv(output_file, sep='\t', index=False)

def save_axis_labels_tsv(all_sample_data, region, output_file):
    """Save axis labels to a TSV file."""
    positions = np.arange(region['start'], region['end'] + 1)
    data_dict = {'Position': positions}
    
    for sample, sample_data in all_sample_data.items():
        data_dict[sample] = [sample_data[pos]["mutations_per_thousand"] for pos in positions]
    
    axis_df = pd.DataFrame(data_dict)
    axis_df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    args = parse_arguments()
    region = parse_region(args.region)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load reference genome
    ref_genome_full = load_reference(args.ref_genome, region['chrom'])
    ref_sequence = ref_genome_full[region['start']-1:region['end']]
    
    # Find motifs
    motif1_results = find_motifs(ref_sequence, args.motif1, region)
    motif2_results = find_motifs(ref_sequence, args.motif2, region)
    
    def process_motifs(results):
        return [(start, end) for strand, start, end in results 
              if start <= region['end'] and end >= region['start']]
    
    motif1_regions = process_motifs(motif1_results)
    motif2_regions = process_motifs(motif2_results)
    
    # Read input TSV
    df = pd.read_csv(args.input, sep='\t', header=None,
                   names=["CHROM", "POS", "REF", "ALT", "SAMPLE", "AD"])
    
    # Process data
    rows = []
    for idx, row in df.iterrows():
        alts = row["ALT"].split(',')
        ad_str = row["AD"]
        try:
            ad = list(map(int, ad_str.split(',')))
        except:
            continue
        
        if len(ad) < 1:
            continue
        
        total_dp = sum(ad)
        if total_dp == 0:
            continue
        
        ref_dp = ad[0]
        for i, alt in enumerate(alts):
            if i + 1 >= len(ad):
                continue
            alt_dp = ad[i+1]
            mutations_per_thousand = (alt_dp / total_dp) * 1000
            if mutations_per_thousand >= args.noise_threshold:
                rows.append({
                    "CHROM": row["CHROM"],
                    "POS": row["POS"],
                    "REF": row["REF"],
                    "ALT": alt,
                    "SAMPLE": row["SAMPLE"],
                    "TOTAL_DP": total_dp,
                    "ALT_DP": alt_dp
                })
    
    new_df = pd.DataFrame(rows)
    
    if new_df.empty:
        print("No data found after filtering!")
        exit(1)
    
    # Получаем список всех образцов
    all_samples = sorted(new_df["SAMPLE"].unique())
    
    # Обрабатываем и строим графики
    all_sample_data = process_and_plot_samples(new_df, region, args, motif1_regions, 
                                             motif2_regions, ref_genome_full, ref_sequence)
    
    if not all_sample_data:
        print("No sample data to process!")
        exit(1)
    
    # Создаем имя для summary файла с именами образцов
    if args.multi_sample and len(all_samples) > 0:
        if len(all_samples) > 3:
            sample_part = f"{len(all_samples)}_samples"
        else:
            sample_part = "_".join(all_samples)
        summary_filename = f"summary_{sample_part}.tsv"
    else:
        summary_filename = "summary.tsv"
    
    # Считаем summary статистику
    summary_data = []
    for sample in all_sample_data.keys():
        sample_df = new_df[(new_df["SAMPLE"] == sample)]
        if not sample_df.empty:
            avg_rate = calculate_average_mutation_rate(sample_df, region, args.noise_threshold, ref_genome_full)
            summary_data.append([sample, avg_rate])
    
    # Сохраняем summary TSV
    summary_output = os.path.join(args.output_dir, summary_filename)
    save_summary_tsv(summary_data, summary_output)
    
    # Сохраняем axis labels TSV если отключены
    if args.disable_axis_labels:
        if args.multi_sample and len(all_samples) > 0:
            if len(all_samples) > 3:
                sample_part = f"{len(all_samples)}_samples"
            else:
                sample_part = "_".join(all_samples)
            axis_filename = f"axis_labels_{sample_part}.tsv"
        else:
            axis_filename = "axis_labels.tsv"
        
        axis_labels_output = os.path.join(args.output_dir, axis_filename)
        save_axis_labels_tsv(all_sample_data, region, axis_labels_output)
    
    print(f"Analysis complete. Results saved to: {args.output_dir}")
    print(f"Summary statistics: {summary_output}")
    if args.disable_axis_labels:
        print(f"Axis labels data: {axis_labels_output}")