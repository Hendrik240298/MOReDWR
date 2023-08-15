/* ---------------------------------------------------------------------
 * The space time finite elements code has been based on Step-3
 * of the deal.II tutorial programs.
 *
 * STEP-3:
 * =======
 *
 * Copyright (C) 1999 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 *
 * ---------------------------------------------------------------------
 * TENSOR-PRODUCT SPACE-TIME FINITE ELEMENTS W/ TIME-SLABBING:
 * ===========================================================
 * Tensor-product space-time code with Q^s finite elements in space and dG-Q^r finite elements in time: cG(s)dG(r)
 * Using Gauss-Legendre quadrature for the temporal quadrature points, since it was shown in [RothThieleKöcherWick2022] that this yields better temporal error estimates
 * The time interval I = (0,T) is being divided into subintervals I_1, I_2, ..., I_M
 * and then instead of solving the PDE all at once on the full space-time domain Q := I x Ω
 * we solve sequentially (forward-in-time) on time slabs Q_n := I_n x Ω to reduce the size of the linear systems.
 * This also allows the usage of different temporal polynomial degrees on each slab, i.e. r := (r_1, r_2, ..., r_M) is now a multiindex.
 *
 * Author: Julian Roth, 2022
 *
 * For more information on tensor-product space-time finite elements please also check out the DTM-project by Uwe Köcher and contributors.
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
// for debugging:
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <numeric>
#include <string>
#include <set>
#include <memory>
#include <map>
#include <sys/stat.h> // for mkdir

using namespace dealii;

void print_as_numpy_arrays_high_resolution(SparseMatrix<double> &matrix,
										   std::ostream &out,
										   const unsigned int precision)
{
	AssertThrow(out.fail() == false, ExcIO());

	out.precision(precision);
	out.setf(std::ios::scientific, std::ios::floatfield);

	std::vector<int> rows;
	std::vector<int> columns;
	std::vector<double> values;
	rows.reserve(matrix.n_nonzero_elements());
	columns.reserve(matrix.n_nonzero_elements());
	values.reserve(matrix.n_nonzero_elements());

	SparseMatrixIterators::Iterator<double, false> it = matrix.begin();
	for (unsigned int i = 0; i < matrix.m(); i++)
	{
		for (it = matrix.begin(i); it != matrix.end(i); ++it)
		{
			rows.push_back(i);
			columns.push_back(it->column());
			values.push_back(matrix.el(i, it->column()));
		}
	}

	for (auto d : values)
		out << d << ' ';
	out << '\n';

	for (auto r : rows)
		out << r << ' ';
	out << '\n';

	for (auto c : columns)
		out << c << ' ';
	out << '\n';
	out << std::flush;

	AssertThrow(out.fail() == false, ExcIO());
}

template <int dim>
class ZeroInitialValues : public Function<dim>
{
public:
	virtual double value(const Point<dim> &p,
						 const unsigned int component = 0) const override;
};

template <int dim>
double ZeroInitialValues<dim>::value(const Point<dim> &p,
									 const unsigned int /*component*/) const
{
	// u_0(x) = 0.
	return 0.0;
}

template <int dim>
class Solution : public Function<dim>
{
public:
	virtual double value(const Point<dim> &p,
						 const unsigned int component = 0) const override;
	virtual Tensor<1, dim>
	gradient(const Point<dim> &p, const unsigned int component = 0) const
		override;
};

template <int dim>
double Solution<dim>::value(const Point<dim> &p,
							const unsigned int /*component*/) const
{
	const double t = this->get_time();
	return 0.; // we don't have a known analytical/manufactured solution
}

template <int dim>
Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p,
									   const unsigned int /*component*/) const
{
	Tensor<1, dim> grad;
	return grad; // we don't have a known analytical/manufactured solution
}

template <int dim>
class RightHandSideRotatingCircle : public Function<dim>
{
public:
	RightHandSideRotatingCircle() : Function<dim>()
	{
	}
	virtual double value(const Point<dim> &p,
						 const unsigned int component = 0) const;
};

template <int dim>
double RightHandSideRotatingCircle<dim>::value(const Point<dim> &p,
											   const unsigned int /*component*/) const
{
	const double t = this->get_time();
	double r = 0.125;
	if (dim == 2)
	{
		// f(t,x,y) = sin(4πt) * indicator_function_circle(center=(x0(t),y0(t)), radius=r)
		// x0(t)    = 1/2 + (1/4)cos(2πt)
		// y0(t)    = 1/2 + (1/4)sin(2πt)
		double x0 = 0.5 + 0.25 * std::cos(2 * M_PI * t);
		double y0 = 0.5 + 0.25 * std::sin(2 * M_PI * t);
		return std::sin(4. * M_PI * t) * (std::pow(p[0] - x0, 2) + std::pow(p[1] - y0, 2) < std::pow(r, 2));
	}
	else
		Assert(false, ExcNotImplemented());
	return -1.0; // to avoid "no return warning"
}

template <int dim>
class RightHandSideSingleSource : public Function<dim>
{
public:
	RightHandSideSingleSource() : Function<dim>()
	{
	}
	virtual double value(const Point<dim> &p,
						 const unsigned int component = 0) const;
};

template <int dim>
double RightHandSideSingleSource<dim>::value(const Point<dim> &p,
											 const unsigned int /*component*/) const
{
	Assert(dim == 1, ExcInternalError());
	const double t = this->get_time();
	// f(t,x) = {0.2, if t in even intervall and x in (1(4,1/2)
	//			{0.0, if t in odd  intervall and x else
	double value = 0;

	if ((p[0] >= (1. / 8.)) && (p[0] <= (3. / 8.)))
		if (int(std::floor(t)) % 2 == 0)
			value = 0.2;
	return value;
}

template <int dim>
class RightHandSideTwoSources : public Function<dim>
{
public:
	RightHandSideTwoSources() : Function<dim>()
	{
	}
	virtual double value(const Point<dim> &p,
						 const unsigned int component = 0) const;
};

template <int dim>
double RightHandSideTwoSources<dim>::value(const Point<dim> &p,
										   const unsigned int /*component*/) const
{
	Assert(dim == 1, ExcInternalError());
	const double t = this->get_time();
	// f(t,x) = {0.2, if t in even intervall and x in (1/8,3/8)
	//			{-0.2, if t in odd intervall and x in (6/8,7/8)
	//			{0.0, else
	double value = 0;

	if ((p[0] >= (1. / 8)) && (p[0] <= (3. / 8)))
		if (int(std::floor(t)) % 2 == 0)
			value = 0.2;
	if ((p[0] >= (6. / 8)) && (p[0] <= (7. / 8)))
		if (int(std::floor(t)) % 2 != 0)
			value = -0.2;
	return value;
}

template <int dim>
class RightHandSideMovingSource : public Function<dim>
{
public:
	RightHandSideMovingSource() : Function<dim>()
	{
	}
	virtual double value(const Point<dim> &p,
						 const unsigned int component = 0) const;
};

template <int dim>
double RightHandSideMovingSource<dim>::value(const Point<dim> &p,
											 const unsigned int /*component*/) const
{
	Assert(dim == 1, ExcInternalError());
	const double t = this->get_time();
	// f(t,x) = {0.2, if t in even intervall and x in (1(4,1/2)
	//			{0.0, if t in odd  intervall and x else
	double value = 0;

	if (t <= 2)
		if (p[0] >= 0.1 + t * 0.4 - 0.05 && p[0] <= 0.1 + t * 0.4 + 0.05)
			if (t <= 1)
				value = 0.2;
			else
				value = -0.5;
	if (t > 2)
		if (p[0] >= 0.9 - (t - 2) * 0.4 - 0.05 && p[0] <= 0.9 - (t - 2) * 0.4 + 0.05)
			if (t <= 3)
				value = 1.0;
			else
				value = -0.75;
	return value;
}

template <int dim>
class DualRightHandSide : public Function<dim>
{
public:
	DualRightHandSide(double _T) : Function<dim>()
	{
		T = _T;
	}
	virtual double value(const Point<dim> &p,
						 const unsigned int component = 0) const;

private:
	double T;
};

template <int dim>
double DualRightHandSide<dim>::value(const Point<dim> &p,
									 const unsigned int /*component*/) const
{
	//	Assert(dim == 1, ExcInternalError());
	//	const double t = this->get_time();
	// f(t,x) = (2/T) * 1_(0,0.5)(x) -> 1_D(x) := { 1 for x in D and 0 else }
	return (2. / T) * (p[0] <= 0.5);
}

class Slab
{
public:
	// constructor
	Slab(unsigned int r, double start_time, double end_time);

	// variables
	Triangulation<1> time_triangulation;
	FE_DGQArbitraryNodes<1> time_fe;
	DoFHandler<1> time_dof_handler;

	double start_time, end_time;
};

Slab::Slab(unsigned int r, double start_time, double end_time) : time_fe(QGauss<1>(r + 1)), time_dof_handler(time_triangulation), start_time(start_time), end_time(end_time)
{
}

template <int dim>
class SpaceTime
{
public:
	SpaceTime(std::string problem_name, unsigned int s, std::vector<unsigned int> r, std::vector<double> time_points = {0., 1.},
			  unsigned int max_n_refinement_cycles = 3, bool refine_space = true, bool refine_time = true, bool split_slabs = true);
	void run();
	void print_grids(std::string file_name_space, std::string file_name_time);
	void print_convergence_table();

private:
	void make_grids();
	void setup_system(std::shared_ptr<Slab> &slab, unsigned int k);
	void assemble_system(std::shared_ptr<Slab> &slab, unsigned int slab_number, unsigned int cycle, bool first_slab, bool assemble_matrix);
	void apply_boundary_conditions(std::shared_ptr<Slab> &slab, unsigned int cycle, bool first_slab);
	void solve(bool invert);
	void output_results(const unsigned int refinement_cycle);
	void output_svg(std::ofstream &out, Vector<double> &space_solution, double time_point, double x_min, double x_max, double y_min, double y_max) const; // only for 1D in space
	void process_solution(std::shared_ptr<Slab> &slab, const unsigned int cycle, bool last_slab);
	void print_coordinates(std::shared_ptr<Slab> &slab, std::string output_dir, unsigned int slab_number);

	std::string problem_name;

	// space
	Triangulation<dim> space_triangulation;
	FE_Q<dim> space_fe;
	DoFHandler<dim> space_dof_handler;

	// time
	std::vector<std::shared_ptr<Slab>> slabs;

	// space-time
	SparsityPattern sparsity_pattern;
	SparseMatrix<double> system_matrix;
	SparseMatrix<double> mass_matrix;
	SparseMatrix<double> jump_matrix;
	SparseDirectUMFPACK A_direct;
	Vector<double> solution;		 // space-time solution on current slab
	Vector<double> last_solution;	 // space-time solution on previous slab
	Vector<double> initial_solution; // u(0) or u(t_0)
	Vector<double> system_rhs;
	Vector<double> dual_rhs;

	double start_time, end_time;
	std::set<std::pair<double, unsigned int>> time_support_points; // (time_support_point, support_point_index)
	unsigned int n_snapshots;
	unsigned int max_n_refinement_cycles;
	bool refine_space, refine_time, split_slabs;
	double L2_error;
	std::vector<double> L2_error_vals;
	unsigned int n_active_time_cells;
	unsigned int n_time_dofs;
	ConvergenceTable convergence_table;
};

template <int dim>
SpaceTime<dim>::SpaceTime(std::string problem_name, unsigned int s, std::vector<unsigned int> r, std::vector<double> time_points,
						  unsigned int max_n_refinement_cycles, bool refine_space, bool refine_time, bool split_slabs) : problem_name(problem_name),
																														 space_fe(s), space_dof_handler(space_triangulation),
																														 max_n_refinement_cycles(max_n_refinement_cycles),
																														 refine_space(refine_space), refine_time(refine_time),
																														 split_slabs(split_slabs)
{
	// time_points = [t_0, t_1, ..., t_M]
	// r = [r_1, r_2, ..., r_M] with r_k is the temporal FE degree on I_k = (t_{k-1},t_k]
	Assert(r.size() + 1 == time_points.size(), ExcDimensionMismatch(r.size() + 1, time_points.size()));

	// create slabs
	for (unsigned int k = 0; k < r.size(); ++k)
		slabs.push_back(std::make_shared<Slab>(r[k], time_points[k], time_points[k + 1]));

	start_time = time_points[0];
	end_time = time_points[time_points.size() - 1];
}

void print_1d_grid(std::ofstream &out, Triangulation<1> &triangulation, double start, double end)
{
	out << "<svg width='1200' height='200' xmlns='http://www.w3.org/2000/svg' version='1.1'>" << std::endl;
	out << "<rect fill='white' width='1200' height='200'/>" << std::endl;
	out << "  <line x1='100' y1='100' x2='1100' y2='100' style='stroke:black;stroke-width:4'/>" << std::endl; // timeline
	out << "  <line x1='100' y1='125' x2='100' y2='75' style='stroke:black;stroke-width:4'/>" << std::endl;	  // first tick

	// ticks
	for (auto &cell : triangulation.active_cell_iterators())
		out << "  <line x1='" << (int)(1000 * (cell->vertex(1)[0] - start) / (end - start)) + 100 << "' y1='125' x2='"
			<< (int)(1000 * (cell->vertex(1)[0] - start) / (end - start)) + 100 << "' y2='75' style='stroke:black;stroke-width:4'/>" << std::endl;

	out << "</svg>" << std::endl;
}

void print_1d_grid_slabwise(std::ofstream &out, std::vector<std::shared_ptr<Slab>> &slabs, double start, double end)
{
	out << "<svg width='1200' height='200' xmlns='http://www.w3.org/2000/svg' version='1.1'>" << std::endl;
	out << "<rect fill='white' width='1200' height='200'/>" << std::endl;
	for (unsigned int k = 0; k < slabs.size(); ++k)
		out << "  <line x1='" << (int)(1000 * (slabs[k]->start_time - start) / (end - start)) + 100 << "' y1='100' x2='"
			<< (int)(1000 * (slabs[k]->end_time - start) / (end - start)) + 100 << "' y2='100' style='stroke:"
			<< ((k % 2) ? "blue" : "black") << ";stroke-width:4'/>" << std::endl; // timeline

	// out << "  <line x1='100' y1='100' x2='1100' y2='100' style='stroke:black;stroke-width:4'/>" << std::endl; // timeline
	out << "  <line x1='100' y1='125' x2='100' y2='75' style='stroke:black;stroke-width:4'/>" << std::endl; // first tick

	// ticks
	for (auto &slab : slabs)
		for (auto &cell : slab->time_triangulation.active_cell_iterators())
			out << "  <line x1='" << (int)(1000 * (cell->vertex(1)[0] - start) / (end - start)) + 100 << "' y1='125' x2='"
				<< (int)(1000 * (cell->vertex(1)[0] - start) / (end - start)) + 100 << "' y2='75' style='stroke:black;stroke-width:4'/>" << std::endl;

	out << "</svg>" << std::endl;
}

template <>
void SpaceTime<2>::print_grids(std::string file_name_space, std::string file_name_time)
{
	// space
	std::ofstream out_space(file_name_space);
	GridOut grid_out_space;
	grid_out_space.write_svg(space_triangulation, out_space);

	// time
	std::ofstream out_time(file_name_time);
	print_1d_grid_slabwise(out_time, slabs, start_time, end_time);
}

template <>
void SpaceTime<1>::print_grids(std::string file_name_space, std::string file_name_time)
{
	// space
	std::ofstream out_space(file_name_space);
	print_1d_grid(out_space, space_triangulation, 0., 1.);

	// time
	std::ofstream out_time(file_name_time);
	print_1d_grid_slabwise(out_time, slabs, start_time, end_time);
}

template <>
void SpaceTime<1>::make_grids()
{
	// create grids
	GridGenerator::hyper_rectangle(space_triangulation, Point<1>(0.), Point<1>(1.));
	for (auto &slab : slabs)
		GridGenerator::hyper_rectangle(slab->time_triangulation, Point<1>(slab->start_time), Point<1>(slab->end_time));

	// ensure that both points of the boundary have boundary_id == 0
	for (auto &cell : space_triangulation.cell_iterators())
		for (unsigned int face = 0; face < GeometryInfo<1>::faces_per_cell; face++)
			if (cell->face(face)->at_boundary())
				cell->face(face)->set_boundary_id(0);

	// globally refine the grids
	space_triangulation.refine_global(5);
	for (auto &slab : slabs)
		slab->time_triangulation.refine_global(0);
}

template <>
void SpaceTime<2>::make_grids()
{
	// Hartmann test problem

	// create grids
	GridGenerator::hyper_rectangle(space_triangulation, Point<2>(0., 0.), Point<2>(1., 1.));
	for (auto &slab : slabs)
		GridGenerator::hyper_rectangle(slab->time_triangulation, Point<1>(slab->start_time), Point<1>(slab->end_time));

	// globally refine the grids
	space_triangulation.refine_global(1);
	for (auto &slab : slabs)
		slab->time_triangulation.refine_global(0);
}

template <int dim>
void SpaceTime<dim>::setup_system(std::shared_ptr<Slab> &slab, unsigned int k)
{
	std::cout << "Slab Q_" << k << " = Ω x (" << slab->start_time << "," << slab->end_time << "):" << std::endl;
	std::cout << "   Finite Elements: cG(" << space_fe.get_degree() << ") (space), dG("
			  << slab->time_fe.get_degree() << ") (time)" << std::endl;
	std::cout << "   Number of active cells: " << space_triangulation.n_active_cells() << " (space), "
			  << slab->time_triangulation.n_active_cells() << " (time)" << std::endl;
	slab->time_dof_handler.distribute_dofs(slab->time_fe);
	std::cout << "   Number of degrees of freedom: " << space_dof_handler.n_dofs() << " (space), "
			  << slab->time_dof_handler.n_dofs() << " (time)" << std::endl;

	if (k == 1)
	{
		/////////////////////////////////////////////////////////////////////////////////////////
		// space-time sparsity pattern = tensor product of spatial and temporal sparsity pattern
		//

		// spatial sparsity pattern
		DynamicSparsityPattern space_dsp(space_dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(space_dof_handler, space_dsp);

		// temporal sparsity pattern
		DynamicSparsityPattern time_dsp(slab->time_dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(slab->time_dof_handler, time_dsp);

		// include jump terms in temporal sparsity pattern
		// for Gauss-Legendre quadrature we need to couple all temporal DoFs between two neighboring time intervals
		unsigned int time_block_size = slab->time_fe.degree + 1;
		for (unsigned int k = 1; k < slab->time_triangulation.n_active_cells(); ++k)
			for (unsigned int ii = 0; ii < time_block_size; ++ii)
				for (unsigned int jj = 0; jj < time_block_size; ++jj)
					time_dsp.add(k * time_block_size + ii, (k - 1) * time_block_size + jj);

		// SparsityPattern time_sparsity_pattern;
		// time_sparsity_pattern.copy_from(time_dsp);
		// std::ofstream out_time_sparsity("time_sparsity_pattern.svg");
		// time_sparsity_pattern.print_svg(out_time_sparsity);

		// space-time sparsity pattern
		DynamicSparsityPattern dsp(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
		for (auto &space_entry : space_dsp)
			for (auto &time_entry : time_dsp)
				dsp.add(
					space_entry.row() + space_dof_handler.n_dofs() * time_entry.row(),		// test  function
					space_entry.column() + space_dof_handler.n_dofs() * time_entry.column() // trial function
				);

		sparsity_pattern.copy_from(dsp);
		std::ofstream out_sparsity("sparsity_pattern.svg");
		sparsity_pattern.print_svg(out_sparsity);

		system_matrix.reinit(sparsity_pattern);
		mass_matrix.reinit(sparsity_pattern);

		if (slab->time_triangulation.n_active_cells() == 1)
			jump_matrix.reinit(sparsity_pattern);
	}
	else
	{
		last_solution.reinit(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
		last_solution = solution;
	}

	solution.reinit(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
	system_rhs.reinit(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
	dual_rhs.reinit(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
}

template <int dim>
void SpaceTime<dim>::assemble_system(std::shared_ptr<Slab> &slab, unsigned int slab_number, unsigned int cycle, bool first_slab, bool assemble_matrix)
{
	// dual rhs
	DualRightHandSide<dim> dual_right_hand_side(end_time - start_time);

	// primal rhs
	std::shared_ptr<Function<dim>> right_hand_side;
	if (problem_name == "single_source")
		right_hand_side = std::make_shared<RightHandSideSingleSource<dim>>();
	else if (problem_name == "two_sources")
		right_hand_side = std::make_shared<RightHandSideTwoSources<dim>>();
	else if (problem_name == "moving_source")
		right_hand_side = std::make_shared<RightHandSideMovingSource<dim>>();
	else if (problem_name == "rotating_circle" && dim == 2)
		right_hand_side = std::make_shared<RightHandSideRotatingCircle<dim>>();
	else if (problem_name == "rotating_circle_long_term" && dim == 2)
		right_hand_side = std::make_shared<RightHandSideRotatingCircle<dim>>();
	else
		Assert(false, ExcNotImplemented());

	// space
	QGauss<dim> space_quad_formula(space_fe.degree + 1);
	FEValues<dim> space_fe_values(space_fe, space_quad_formula,
								  update_values | update_gradients | update_quadrature_points | update_JxW_values);
	const unsigned int space_dofs_per_cell = space_fe.n_dofs_per_cell();
	std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);

	// time
	QGauss<1> time_quad_formula(slab->time_fe.degree + 1);
	FEValues<1> time_fe_values(slab->time_fe, time_quad_formula,
							   update_values | update_gradients | update_quadrature_points | update_JxW_values);
	const unsigned int time_dofs_per_cell = slab->time_fe.n_dofs_per_cell();
	std::vector<types::global_dof_index> time_local_dof_indices(time_dofs_per_cell);
	std::vector<types::global_dof_index> time_prev_local_dof_indices(time_dofs_per_cell);

	// time FEValues for t_m^+ on current time interval I_m
	FEValues<1> time_fe_face_values(slab->time_fe, Quadrature<1>({Point<1>(0.)}), update_values); // using left box rule quadrature
	// time FEValues for t_m^- on previous time interval I_{m-1}
	FEValues<1> time_prev_fe_face_values(slab->time_fe, Quadrature<1>({Point<1>(1.)}), update_values); // using right box rule quadrature

	// local contributions on space-time cell
	FullMatrix<double> cell_matrix(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
	FullMatrix<double> cell_mass(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
	FullMatrix<double> cell_jump(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
	Vector<double> cell_rhs(space_dofs_per_cell * time_dofs_per_cell);
	Vector<double> cell_dual_rhs(space_dofs_per_cell * time_dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices(space_dofs_per_cell * time_dofs_per_cell);

	// locally assemble on each space-time cell
	for (const auto &space_cell : space_dof_handler.active_cell_iterators())
	{
		space_fe_values.reinit(space_cell);
		space_cell->get_dof_indices(space_local_dof_indices);
		for (const auto &time_cell : slab->time_dof_handler.active_cell_iterators())
		{
			time_fe_values.reinit(time_cell);
			time_cell->get_dof_indices(time_local_dof_indices);

			cell_matrix = 0;
			cell_mass = 0;
			cell_rhs = 0;
			cell_dual_rhs = 0;
			cell_jump = 0;

			for (const unsigned int qq : time_fe_values.quadrature_point_indices())
			{
				// time quadrature point
				const double t_qq = time_fe_values.quadrature_point(qq)[0];
				right_hand_side->set_time(t_qq);
				dual_right_hand_side.set_time(t_qq);

				for (const unsigned int q : space_fe_values.quadrature_point_indices())
				{
					// space quadrature point
					const auto x_q = space_fe_values.quadrature_point(q);

					for (const unsigned int i : space_fe_values.dof_indices())
						for (const unsigned int ii : time_fe_values.dof_indices())
						{
							// right hand side
							cell_rhs(i + ii * space_dofs_per_cell) += (space_fe_values.shape_value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ_{i,ii}(t_qq, x_q)
																	   right_hand_side->value(x_q) *											// f(t_qq, x_q)
																	   space_fe_values.JxW(q) * time_fe_values.JxW(qq)							// d(t,x)
							);

							// dual right hand side
							cell_dual_rhs(i + ii * space_dofs_per_cell) += (space_fe_values.shape_value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ_{i,ii}(t_qq, x_q)
																			dual_right_hand_side.value(x_q) *										 // f_dual(t_qq, x_q)
																			space_fe_values.JxW(q) * time_fe_values.JxW(qq)							 // d(t,x)
							);

							// system matrix
							if (assemble_matrix)
								for (const unsigned int j : space_fe_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
									{
										cell_matrix(
											j + jj * space_dofs_per_cell,
											i + ii * space_dofs_per_cell) += (space_fe_values.shape_value(i, q) * time_fe_values.shape_grad(ii, qq)[0] *  // ∂_t ϕ_{i,ii}(t_qq, x_q)
																				  space_fe_values.shape_value(j, q) * time_fe_values.shape_value(jj, qq)  //     ϕ_{j,jj}(t_qq, x_q)
																																						  // +
																			  + space_fe_values.shape_grad(i, q) * time_fe_values.shape_value(ii, qq) *	  // ∇_x ϕ_{i,ii}(t_qq, x_q)
																					space_fe_values.shape_grad(j, q) * time_fe_values.shape_value(jj, qq) // ∇_x ϕ_{j,jj}(t_qq, x_q)
																			  ) *
																			 space_fe_values.JxW(q) * time_fe_values.JxW(qq); // d(t,x)

										if (time_cell->active_cell_index() == 0)
										{
											cell_mass(
												j + jj * space_dofs_per_cell,
												i + ii * space_dofs_per_cell) += (space_fe_values.shape_value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ_{i,ii}(t_qq, x_q)
																				  space_fe_values.shape_value(j, q) * time_fe_values.shape_value(jj, qq)   // ϕ_{j,jj}(t_qq, x_q)
																				  ) *
																				 space_fe_values.JxW(q) * time_fe_values.JxW(qq); // d(t,x)
										}
									}
						}
				}
			}

			// assemble jump terms in system matrix and intial condition in RHS
			// jump terms: ([u]_m,φ_m^+)_Ω = (u_m^+,φ_m^+)_Ω - (u_m^-,φ_m^+)_Ω = (A) - (B)
			time_fe_face_values.reinit(time_cell);

			// first we assemble (A): (u_m^+,φ_m^+)_Ω
			if (assemble_matrix)
				for (const unsigned int q : space_fe_values.quadrature_point_indices())
					for (const unsigned int i : space_fe_values.dof_indices())
						for (const unsigned int ii : time_fe_values.dof_indices())
							for (const unsigned int j : space_fe_values.dof_indices())
								for (const unsigned int jj : time_fe_values.dof_indices())
									cell_matrix(
										j + jj * space_dofs_per_cell,
										i + ii * space_dofs_per_cell) += (space_fe_values.shape_value(i, q) * time_fe_face_values.shape_value(ii, 0) * //  ϕ_{i,ii}(t_m^+, x_q)
																		  space_fe_values.shape_value(j, q) * time_fe_face_values.shape_value(jj, 0)   //  ϕ_{j,jj}(t_m^+, x_q)
																		  ) *
																		 space_fe_values.JxW(q); //  d(x)

			// initial condition and jump terms
			if (time_cell->active_cell_index() == 0)
			{
				//////////////////////////
				// initial condition

				// (u_0^-,φ_0^-)_Ω
				if (first_slab)
					for (const unsigned int q : space_fe_values.quadrature_point_indices())
					{
						double initial_solution_x_q = 0.;
						for (const unsigned int j : space_fe_values.dof_indices())
						{
							initial_solution_x_q += initial_solution[space_local_dof_indices[j]] * space_fe_values.shape_value(j, q);
						}

						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
							{
								cell_rhs(i + ii * space_dofs_per_cell) += (initial_solution_x_q *														// u0(x_q)
																		   space_fe_values.shape_value(i, q) * time_fe_face_values.shape_value(ii, 0) * // ϕ_{i,ii}(0^+, x_q)
																		   space_fe_values.JxW(q)														// d(x)
								);
							}
					}
			}
			else
			{
				//////////////
				// jump term

				// now we assemble (B): - (u_m^-,φ_m^+)_Ω
				// NOTE: cell_jump is a space-time cell matrix because we are using Gauss-Legendre quadrature in time
				if (assemble_matrix)
					for (const unsigned int q : space_fe_values.quadrature_point_indices())
						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
								for (const unsigned int j : space_fe_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
										cell_jump(
											j + jj * space_dofs_per_cell,
											i + ii * space_dofs_per_cell) += (-1. * space_fe_values.shape_value(i, q) * time_prev_fe_face_values.shape_value(ii, 0) * // -ϕ_{i,ii}(t_m^-, x_q)
																			  space_fe_values.shape_value(j, q) * time_fe_face_values.shape_value(jj, 0)			  //  ϕ_{j,jj}(t_m^+, x_q)
																			  ) *
																			 space_fe_values.JxW(q); //  d(x)
			}

			if (assemble_matrix && (slab->time_triangulation.n_active_cells() == 1))
			{
				time_prev_fe_face_values.reinit(time_cell);
				time_cell->get_dof_indices(time_prev_local_dof_indices);

				//////////////
				// jump term

				// now we assemble (B): - (u_m^-,φ_m^+)_Ω
				// NOTE: cell_jump is a space-time cell matrix because we are using Gauss-Legendre quadrature in time
				for (const unsigned int q : space_fe_values.quadrature_point_indices())
					for (const unsigned int i : space_fe_values.dof_indices())
						for (const unsigned int ii : time_fe_values.dof_indices())
							for (const unsigned int j : space_fe_values.dof_indices())
								for (const unsigned int jj : time_fe_values.dof_indices())
									cell_jump(
										j + jj * space_dofs_per_cell,
										i + ii * space_dofs_per_cell) += (-1. * space_fe_values.shape_value(i, q) * time_prev_fe_face_values.shape_value(ii, 0) * // -ϕ_{i,ii}(t_m^-, x_q)
																		  space_fe_values.shape_value(j, q) * time_fe_face_values.shape_value(jj, 0)			  //  ϕ_{j,jj}(t_m^+, x_q)
																		  ) *
																		 space_fe_values.JxW(q); //  d(x)
			}

			// distribute local to global
			for (const unsigned int i : space_fe_values.dof_indices())
				for (const unsigned int ii : time_fe_values.dof_indices())
				{
					// right hand side
					system_rhs(space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs()) += cell_rhs(i + ii * space_dofs_per_cell);

					// dual right hand side
					dual_rhs(space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs()) += cell_dual_rhs(i + ii * space_dofs_per_cell);

					// system matrix
					if (assemble_matrix)
						for (const unsigned int j : space_fe_values.dof_indices())
							for (const unsigned int jj : time_fe_values.dof_indices())
								system_matrix.add(
									space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs(),
									space_local_dof_indices[j] + time_local_dof_indices[jj] * space_dof_handler.n_dofs(),
									cell_matrix(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));
				}

			// distribute cell jump
			if (assemble_matrix && (time_cell->active_cell_index() > 0))
				for (const unsigned int i : space_fe_values.dof_indices())
					for (const unsigned int ii : time_fe_values.dof_indices())
						for (const unsigned int j : space_fe_values.dof_indices())
							for (const unsigned int jj : time_fe_values.dof_indices())
								system_matrix.add(
									space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs(),
									space_local_dof_indices[j] + time_prev_local_dof_indices[jj] * space_dof_handler.n_dofs(),
									cell_jump(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));

			if (assemble_matrix && (time_cell->active_cell_index() == 0))
				for (const unsigned int i : space_fe_values.dof_indices())
					for (const unsigned int ii : time_fe_values.dof_indices())
						for (const unsigned int j : space_fe_values.dof_indices())
							for (const unsigned int jj : time_fe_values.dof_indices())
							{
								jump_matrix.add(
									space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs(),
									space_local_dof_indices[j] + time_prev_local_dof_indices[jj] * space_dof_handler.n_dofs(),
									cell_jump(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));

								mass_matrix.add(
									space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs(),
									space_local_dof_indices[j] + time_local_dof_indices[jj] * space_dof_handler.n_dofs(),
									cell_mass(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));
							}

			// prepare next time cell
			if (time_cell->active_cell_index() < slab->time_triangulation.n_active_cells() - 1)
			{
				time_prev_fe_face_values.reinit(time_cell);
				time_cell->get_dof_indices(time_prev_local_dof_indices);
			}
		}
	}

	std::string output_dir = "../../../Data/" + std::to_string(dim) + "D/" + problem_name + "/slabwise/FOM/cycle=" + std::to_string(cycle) + "-0/"; // HF: set time cycle

	////////////////////////////////////////////////////
	// save system & jump matrix and rhs to file (NO BC)
	if (assemble_matrix)
	{
		std::ofstream matrix_no_bc_out(output_dir + "matrix_no_bc.txt");
		print_as_numpy_arrays_high_resolution(system_matrix, matrix_no_bc_out, /*precision*/ 16);

		std::ofstream jump_matrix_no_bc_out(output_dir + "jump_matrix_no_bc.txt");
		print_as_numpy_arrays_high_resolution(jump_matrix, jump_matrix_no_bc_out, /*precision*/ 16);

		std::ofstream mass_matrix_no_bc_out(output_dir + "mass_matrix_no_bc.txt");
		print_as_numpy_arrays_high_resolution(mass_matrix, mass_matrix_no_bc_out, /*precision*/ 16);
	}

	std::ofstream rhs_no_bc_out(output_dir + "rhs_no_bc_" + Utilities::int_to_string(slab_number, 5) + ".txt");
	system_rhs.print(rhs_no_bc_out, /*precision*/ 16);

	std::ofstream dual_rhs_no_bc_out(output_dir + "dual_rhs_no_bc_" + Utilities::int_to_string(slab_number, 5) + ".txt");
	dual_rhs.print(dual_rhs_no_bc_out, /*precision*/ 16);

	if (!first_slab)
	{
		Vector<double> tmp(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
		jump_matrix.vmult(tmp, last_solution);
		system_rhs -= tmp;
	}

	/////////////////////////////////////
	// apply BC to linear equation system
	apply_boundary_conditions(slab, cycle, first_slab);
}

template <int dim>
void SpaceTime<dim>::apply_boundary_conditions(std::shared_ptr<Slab> &slab, unsigned int cycle, bool first_slab)
{
	// apply the spatial Dirichlet boundary conditions at each temporal DoF
	Functions::ZeroFunction<dim> solution_func; // NOTE: using homogeneous Dirichlet boundary conditions

	// list of constrained space-time DoF indices
	std::vector<types::global_dof_index> boundary_ids;

	FEValues<1> time_fe_values(slab->time_fe, Quadrature<1>(slab->time_fe.get_unit_support_points()), update_quadrature_points);
	std::vector<types::global_dof_index> time_local_dof_indices(slab->time_fe.n_dofs_per_cell());

	for (const auto &time_cell : slab->time_dof_handler.active_cell_iterators())
	{
		time_fe_values.reinit(time_cell);
		time_cell->get_dof_indices(time_local_dof_indices);

		// using temporal support points as quadrature points
		for (const unsigned int qq : time_fe_values.quadrature_point_indices())
		{
			// time quadrature point
			double t_qq = time_fe_values.quadrature_point(qq)[0];
			solution_func.set_time(t_qq);
			time_support_points.insert(std::make_pair(t_qq, time_local_dof_indices[qq]));

			// determine spatial boundary values at temporal support point
			std::map<types::global_dof_index, double> boundary_values;
			VectorTools::interpolate_boundary_values(space_dof_handler, 0, solution_func, boundary_values);

			// calculate the correct space-time entry and apply the Dirichlet BC
			for (auto &entry : boundary_values)
			{
				types::global_dof_index id = entry.first + time_local_dof_indices[qq] * space_dof_handler.n_dofs();

				// apply BC
				for (typename SparseMatrix<double>::iterator p = system_matrix.begin(id); p != system_matrix.end(id); ++p)
					p->value() = 0.;
				system_matrix.set(id, id, 1.);
				system_rhs(id) = entry.second;
				boundary_ids.push_back(id);
			}
		}
	}

	if (first_slab)
	{
		std::ofstream boundary_id_out("../../../Data/" + std::to_string(dim) + "D/" + problem_name + "/slabwise/FOM/cycle=" + std::to_string(cycle) + "-0/boundary_id.txt"); // HF: set time cycle
		for (unsigned int i = 0; i < boundary_ids.size() - 1; ++i)
			boundary_id_out << boundary_ids[i] << " ";
		boundary_id_out << boundary_ids[boundary_ids.size() - 1];
	}
}

template <int dim>
void SpaceTime<dim>::solve(bool invert)
{
	if (invert)
		A_direct.initialize(system_matrix);
	A_direct.vmult(solution, system_rhs);
}

// To avoid using external plotting software, we create the 1D output in the form of an SVG file ourselves.
// cG(1):            we use SVG's "line" to visualize solution
// cG(2), cG(3):     we use SVG's "path" to visualize solution with quadratic/cubic Bezier curves
// cG(s) with s > 3: we use a sufficiently high number of support points in the spatial cell and interpolate linearly
template <>
void SpaceTime<1>::output_svg(std::ofstream &out, Vector<double> &space_solution, double time_point, double x_min, double x_max, double y_min, double y_max) const
{
	// create an equidistant quadrature and FEValues object
	const unsigned int n_points_per_cell = (space_fe.get_degree() <= 3) ? 2 : std::pow(2, space_fe.get_degree()) + 4;
	QIterated<1> space_quad(QTrapez<1>(), n_points_per_cell - 1); // need to use QTrapezoid for deal.II 9.3.0
	FEValues<1> space_fe_values(space_fe, space_quad, update_values | update_gradients | update_quadrature_points);

	// create 2D vectors to store the quadrature points and the solution at these points
	std::vector<std::vector<Point<1>>> q_points(space_triangulation.n_active_cells(), std::vector<Point<1>>(n_points_per_cell, Point<1>(0.)));
	std::vector<std::vector<double>> u_values(space_triangulation.n_active_cells(), std::vector<double>(n_points_per_cell));
	// for the quadratic and cubic Bezier curves, we also need the gradients at the support points
	std::vector<std::vector<Tensor<1, 1>>> u_gradients(space_triangulation.n_active_cells(), std::vector<Tensor<1, 1>>(n_points_per_cell));

	for (const auto &space_cell : space_dof_handler.active_cell_iterators())
	{
		unsigned int i = space_cell->index();
		space_fe_values.reinit(space_cell);

		space_fe_values.get_function_values(space_solution, u_values[i]);
		if (space_fe.get_degree() == 2 || space_fe.get_degree() == 3)
			space_fe_values.get_function_gradients(space_solution, u_gradients[i]);
		q_points[i] = space_fe_values.get_quadrature_points();
	}

	// convert x and y values into coordinates of the canvas
	// note: the coordinates (0,0) are located in the upper left corner
	auto x_coord = [&](double x)
	{
		return (1000 * (x - x_min) / (x_max - x_min)) + 50;
	};

	auto y_coord = [&](double y)
	{
		return 1150 - 1000 * (y - y_min) / (y_max - y_min);
	};

	// print out the spatial grid with ticks for the edges of the spatial cells and the solution
	out << "<svg width='1200' height='1200' xmlns='http://www.w3.org/2000/svg' version='1.1'>" << std::endl;
	out << "<rect fill='white' width='1200' height='1200'/>" << std::endl;
	out << "  <text x='500' y='50' font-size='4em'>t = " << time_point << "</text>" << std::endl;												 // title
	out << "  <line x1='50' y1='" << y_coord(0.) << "' x2='1150' y2='" << y_coord(0.) << "' style='stroke:black;stroke-width:4'/>" << std::endl; // x-axis (horizontal)
	out << "  <line x1='50' y1='50' x2='50' y2='1150' style='stroke:black;stroke-width:4'/>" << std::endl;										 // y-axis (vertical)

	// ticks
	for (auto &space_cell : space_triangulation.active_cell_iterators())
		out << "  <line x1='" << x_coord(space_cell->vertex(1)[0]) << "' y1='" << y_coord(0.) + 25 << "' x2='"
			<< x_coord(space_cell->vertex(1)[0]) << "' y2='" << y_coord(0.) - 25 << "' style='stroke:black;stroke-width:4'/>" << std::endl;

	// arrow of x-axis
	out << "  <line x1='1125' y1='" << y_coord(0.) + 25 << "' x2='1150' y2='" << y_coord(0.) << "' style='stroke:black;stroke-width:4'/>" << std::endl;
	out << "  <line x1='1125' y1='" << y_coord(0.) - 25 << "' x2='1150' y2='" << y_coord(0.) << "' style='stroke:black;stroke-width:4'/>" << std::endl;

	// arrow of y-axis
	out << "  <line x1='75' y1='75' x2='50' y2='50' style='stroke:black;stroke-width:4'/>" << std::endl;
	out << "  <line x1='25' y1='75' x2='50' y2='50' style='stroke:black;stroke-width:4'/>" << std::endl
		<< std::endl;

	// plot the finite element solution
	if (space_fe.get_degree() == 2)
	{
		// use quadratic Bezier curves to visualize the solution of the quadratic FE
		for (const auto &space_cell : space_dof_handler.active_cell_iterators())
		{
			unsigned int i = space_cell->index();
			// to construct a quadratic Bezier curve of a parabola p on the interval [a,b] we need:
			// 1.) the value at the left boundary:      p(a)
			// 2.) the value at the right boundary:     p(b)
			// 3.) the gradient at the left boundary:  p'(a)
			// 4.) the gradient at the right boundary: p'(b)
			//
			// we then have the following three control points for the Bezier curve:
			// 1.) left boundary: (a, p(a))
			// 2.) right boundary: (b, p(b))
			// 3.) the intersection of the tangents to the parabola at the left and right boundary: (c, p'(a)*(c-a)+p(a))
			// solving for c we then get c = [p(a)-p(b)-a*p'(a)+b*p'(b)] / [p'(b)-p'(a)] = (a+b)/2
			double a = q_points[i][0][0];
			double b = q_points[i][1][0];
			double val_a = u_values[i][0];
			double val_b = u_values[i][1];
			double grad_a = u_gradients[i][0][0];

			// compute the intersection of the tangents
			double c = 0.5 * (a + b);
			double val_c = grad_a * (c - a) + val_a;

			out << "  <path d='M" << x_coord(a) << "," << y_coord(val_a) << " Q" << x_coord(c) << "," << y_coord(val_c) << " "
				<< x_coord(b) << "," << y_coord(val_b) << "' style='stroke:blue;stroke-width:2' fill='transparent'/>" << std::endl;
		}
	}
	else if (space_fe.get_degree() == 3)
	{
		// use cubic Bezier curves to visualize the solution of the cubic FE
		for (const auto &space_cell : space_dof_handler.active_cell_iterators())
		{
			unsigned int i = space_cell->index();
			// to construct a cubic Bezier curve of a parabola p on the interval [a,b] we need:
			// 1.) the value at the left boundary:      p(a)
			// 2.) the value at the right boundary:     p(b)
			// 3.) the gradient at the left boundary:  p'(a)
			// 4.) the gradient at the right boundary: p'(b)
			//
			// link on cubic Bezier interpolation: https://web.archive.org/web/20131225210855/http://people.sc.fsu.edu/~jburkardt/html/bezier_interpolation.html
			// we then have the following four control points for the Bezier curve:
			// 1.) left boundary: (a, p(a))
			// 2.) right boundary: (b, p(b))
			// 3.) a point on the tangent to the left boundary: (c, p'(a)*(c-a)+p(a)) with c = (2a+b)/3
			// 4.) a point on the tangent to the right boundary: (d, p'(b)*(d-b)+p(b)) with b = (a+2b)/3
			double a = q_points[i][0][0];
			double b = q_points[i][1][0];
			double val_a = u_values[i][0];
			double val_b = u_values[i][1];
			double grad_a = u_gradients[i][0][0];
			double grad_b = u_gradients[i][1][0];

			// compute the other two control points
			double c = (2. * a + b) / 3.;
			double val_c = grad_a * (c - a) + val_a;
			double d = (a + 2. * b) / 3.;
			double val_d = grad_b * (d - b) + val_b;

			out << "  <path d='M" << x_coord(a) << "," << y_coord(val_a) << " C" << x_coord(c) << "," << y_coord(val_c) << " " << x_coord(d) << "," << y_coord(val_d) << " "
				<< x_coord(b) << "," << y_coord(val_b) << "' style='stroke:blue;stroke-width:2' fill='transparent'/>" << std::endl;
		}
	}
	else
	{
		// use a linear interpolation to visualize the solution
		// for cG(s) with s > 3 we use many quadrature points
		for (unsigned int i = 0; i < space_triangulation.n_active_cells(); ++i)
			for (unsigned int j = 0; j < n_points_per_cell - 1; ++j)
				out << "  <line x1='" << x_coord(q_points[i][j][0]) << "' y1='" << y_coord(u_values[i][j]) << "' x2='" << x_coord(q_points[i][j + 1][0]) << "' y2='" << y_coord(u_values[i][j + 1]) << "' style='stroke:blue;stroke-width:2'/>" << std::endl;
	}

	out << "</svg>" << std::endl;
}

template <>
void SpaceTime<1>::output_results(const unsigned int refinement_cycle)
{
	std::string output_dir = "../../../Data/1D/" + problem_name + "/slabwise/FOM/cycle=" + std::to_string(refinement_cycle) + "-0/"; // HF: set time cycle

	// highest and lowest value of solution (hardcoded for the given 1D problem)
	double y_max = 0.009;  // 1.4; //std::max(*std::max_element(solution.begin(), solution.end()), 0.);
	double y_min = -0.004; // 0.0; //std::min(*std::min_element(solution.begin(), solution.end()), 0.);

	// left and right boundary of space_triangulation
	double x_min = 0.;
	double x_max = 1.;

	// output results as SVG files
	for (auto time_point : time_support_points)
	{
		double t_qq = time_point.first;
		unsigned int ii = time_point.second;

		Vector<double> space_solution(space_dof_handler.n_dofs());
		for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
			space_solution(i) = solution(i + ii * space_dof_handler.n_dofs());

		std::ofstream output(output_dir + "solution" + Utilities::int_to_string(n_snapshots, 5) + ".svg");
		output_svg(output, space_solution, t_qq, x_min, x_max, y_min, y_max);

		n_snapshots++;
	}
}

template <>
void SpaceTime<2>::output_results(const unsigned int refinement_cycle)
{
	std::string output_dir = "../../../Data/2D/" + problem_name + "/slabwise/FOM/cycle=" + std::to_string(refinement_cycle) + "-0/"; // HF: set time cycle

	// output results as VTK files
	for (auto time_point : time_support_points)
	{
		double t_qq = time_point.first;
		unsigned int ii = time_point.second;

		DataOut<2> data_out;
		data_out.attach_dof_handler(space_dof_handler);

		Vector<double> space_solution(space_dof_handler.n_dofs());
		for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
			space_solution(i) = solution(i + ii * space_dof_handler.n_dofs());

		data_out.add_data_vector(space_solution, "u");
		data_out.build_patches();

		data_out.set_flags(DataOutBase::VtkFlags(t_qq, ii));

		std::ofstream output(output_dir + "solution" + Utilities::int_to_string(n_snapshots, 5) + ".vtk");
		data_out.write_vtk(output);

		n_snapshots++;
	}
}

template <int dim>
void SpaceTime<dim>::process_solution(std::shared_ptr<Slab> &slab, const unsigned int cycle, bool last_slab)
{
	Solution<dim> solution_func;
	double L2_error = 0.;

	FEValues<1> time_fe_values(slab->time_fe, QGauss<1>(slab->time_fe.degree + 2), update_values | update_quadrature_points | update_JxW_values);
	std::vector<types::global_dof_index> time_local_dof_indices(slab->time_fe.n_dofs_per_cell());

	for (const auto &time_cell : slab->time_dof_handler.active_cell_iterators())
	{
		time_fe_values.reinit(time_cell);
		time_cell->get_dof_indices(time_local_dof_indices);

		for (const unsigned int qq : time_fe_values.quadrature_point_indices())
		{
			// time quadrature point
			double t_qq = time_fe_values.quadrature_point(qq)[0];
			solution_func.set_time(t_qq);

			// get the space solution at the quadrature point
			Vector<double> space_solution(space_dof_handler.n_dofs());
			for (const unsigned int ii : time_fe_values.dof_indices())
			{
				for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
					space_solution(i) += solution(i + time_local_dof_indices[ii] * space_dof_handler.n_dofs()) * time_fe_values.shape_value(ii, qq);
			}

			// compute the L2 error at the temporal quadrature point
			Vector<float> difference_per_cell(space_triangulation.n_active_cells());
			VectorTools::integrate_difference(space_dof_handler, space_solution, solution_func,
											  difference_per_cell, QGauss<dim>(space_fe.degree + 2),
											  VectorTools::L2_norm);
			const double L2_error_t_qq = VectorTools::compute_global_error(space_triangulation,
																		   difference_per_cell, VectorTools::L2_norm);

			// add local contributions to global L2 error
			L2_error += std::pow(L2_error_t_qq, 2) * time_fe_values.JxW(qq);
		}
	}

	n_active_time_cells += slab->time_triangulation.n_active_cells();
	n_time_dofs += slab->time_dof_handler.n_dofs();

	if (last_slab)
	{
		L2_error = std::sqrt(L2_error);
		L2_error_vals.push_back(L2_error);

		// add values to
		const unsigned int n_active_cells = space_triangulation.n_active_cells() * n_active_time_cells;
		const unsigned int n_space_dofs = space_dof_handler.n_dofs();
		const unsigned int n_dofs = n_space_dofs * n_time_dofs;

		convergence_table.add_value("cycle", cycle);
		convergence_table.add_value("cells", n_active_cells);
		convergence_table.add_value("dofs", n_dofs);
		convergence_table.add_value("dofs(space)", n_space_dofs);
		convergence_table.add_value("dofs(time)", n_time_dofs);
		convergence_table.add_value("L2", L2_error);
	}
}

template <int dim>
void SpaceTime<dim>::print_convergence_table()
{
	convergence_table.set_precision("L2", 3);
	convergence_table.set_scientific("L2", true);
	convergence_table.set_tex_caption("cells", "\\# cells");
	convergence_table.set_tex_caption("dofs", "\\# dofs");
	convergence_table.set_tex_caption("dofs(space)", "\\# dofs space");
	convergence_table.set_tex_caption("dofs(time)", "\\# dofs time");
	convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
	convergence_table.set_tex_format("cells", "r");
	convergence_table.set_tex_format("dofs", "r");
	convergence_table.set_tex_format("dofs(space)", "r");
	convergence_table.set_tex_format("dofs(time)", "r");
	std::cout << std::endl;
	convergence_table.write_text(std::cout);

	convergence_table.add_column_to_supercolumn("cycle", "n cells");
	convergence_table.add_column_to_supercolumn("cells", "n cells");
	std::vector<std::string> new_order;
	new_order.emplace_back("n cells");
	new_order.emplace_back("L2");
	convergence_table.set_column_order(new_order);
	if (refine_space && refine_time)
		convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);

	// compute convergence rates from 3 consecutive errors
	for (unsigned int i = 0; i < L2_error_vals.size(); ++i)
	{
		if (i < 2)
			convergence_table.add_value("L2...", std::string("-"));
		else
		{
			double p0 = L2_error_vals[i - 2];
			double p1 = L2_error_vals[i - 1];
			double p2 = L2_error_vals[i];
			convergence_table.add_value("L2...", std::log(std::fabs((p0 - p1) / (p1 - p2))) / std::log(2.));
		}
	}

	std::cout << std::endl;
	convergence_table.write_text(std::cout);
}

template <int dim>
void SpaceTime<dim>::print_coordinates(std::shared_ptr<Slab> &slab, std::string output_dir, unsigned int slab_number)
{
	// space
	std::vector<Point<dim>> space_support_points(space_dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(
		MappingQ1<dim, dim>(),
		space_dof_handler,
		space_support_points);

	std::ofstream x_out(output_dir + "coordinates_x.txt");
	x_out.precision(9);
	x_out.setf(std::ios::scientific, std::ios::floatfield);

	for (int d = 0; d < dim; d++)
	{
		for (auto point : space_support_points)
			x_out << point[d] << ' ';
		x_out << std::endl;
	}

	// time
	std::vector<Point<1>> time_support_points(slab->time_dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(
		MappingQ1<1, 1>(),
		slab->time_dof_handler,
		time_support_points);

	std::ofstream t_out(output_dir + "coordinates_t_" + Utilities::int_to_string(slab_number, 5) + ".txt");
	t_out.precision(9);
	t_out.setf(std::ios::scientific, std::ios::floatfield);

	for (auto point : time_support_points)
		t_out << point[0] << ' ';
	t_out << std::endl;
}

template <int dim>
void SpaceTime<dim>::run()
{
	std::cout << "Starting problem " << problem_name << " in " << dim << "+1D..." << std::endl;

	// create a coarse grid
	make_grids();

	// Refinement loop
	for (unsigned int cycle = 0; cycle < max_n_refinement_cycles; ++cycle)
	{
		std::cout
			<< "-------------------------------------------------------------"
			<< std::endl;
		std::cout << "|                REFINEMENT CYCLE: " << cycle;
		std::cout << "                         |" << std::endl;
		std::cout
			<< "-------------------------------------------------------------"
			<< std::endl;

		// create output directory if necessary
		std::string dim_dir = "../../../Data/" + std::to_string(dim) + "D/";
		std::string problem_dir = dim_dir + problem_name + "/";
		std::string slabwise_dir = problem_dir + "slabwise/";
		std::string output_dir_tmp = slabwise_dir + "FOM/";
		std::string output_dir = slabwise_dir + "FOM/" + "cycle=" + std::to_string(cycle) + "-0/"; // HF: set time cycle
		std::cout << output_dir << std::endl;
		for (auto dir : {"../../../Data/", dim_dir.c_str(), problem_dir.c_str(), slabwise_dir.c_str(), output_dir_tmp.c_str(), output_dir.c_str()})
			mkdir(dir, S_IRWXU);

		// reset values from last refinement cycle
		n_snapshots = 0;
		n_active_time_cells = 0;
		n_time_dofs = 0;
		L2_error = 0.;

		////////////////////////////////////////////
		// initial value: u(0)
		//
		space_dof_handler.distribute_dofs(space_fe);

		// no hanging node constraints
		AffineConstraints<double> constraints;
		constraints.close();

		// compute initial value vector -> NOTE: zero initial conditions
		initial_solution.reinit(space_dof_handler.n_dofs());
		VectorTools::project(space_dof_handler,
							 constraints,
							 QGauss<dim>(space_fe.degree + 1),
							 ZeroInitialValues<dim>(),
							 initial_solution);

		for (unsigned int k = 0; k < slabs.size(); ++k)
		{
			// create and solve linear system
			setup_system(slabs[k], k + 1);
			assemble_system(slabs[k], k, cycle, k == 0, k == 0);
			solve(k == 0);

			// output Space-Time DoF coordinates
			print_coordinates(slabs[k], output_dir, k);

			// output solution to txt file
			std::ofstream solution_out(output_dir + "solution_" + Utilities::int_to_string(k, 5) + ".txt");
			solution.print(solution_out, /*precision*/ 16);

			// output results as SVG or VTK files
			output_results(cycle);

			// Compute the error to the analytical solution
			process_solution(slabs[k], cycle, (k == slabs.size() - 1));

			///////////////////////
			// prepare next slab
			//

			// evaluate solution at end of slab, i.e. on time inteval end of the last time interval, as initial value for next slab
			FEValues<1> time_fe_values(slabs[k]->time_fe, Quadrature<1>({Point<1>(1.)}), update_values);
			auto cell_time = slabs[k]->time_dof_handler.begin();
			while (std::next(cell_time) != slabs[k]->time_dof_handler.end())
				cell_time++;
			time_fe_values.reinit(cell_time);

			std::vector<types::global_dof_index> local_dof_indices(slabs[k]->time_fe.dofs_per_cell);
			cell_time->get_dof_indices(local_dof_indices);

			initial_solution = 0.;
			for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
				for (const unsigned int ii : time_fe_values.dof_indices())
					initial_solution(i) += solution(i + local_dof_indices[ii] * space_dof_handler.n_dofs()) * time_fe_values.shape_value(ii, 0);

			// remove old temporal support points
			time_support_points.clear();
		}

		// refine mesh
		if (cycle < max_n_refinement_cycles - 1)
		{
			space_triangulation.refine_global(refine_space);

			if (split_slabs)
			{
				std::vector<std::shared_ptr<Slab>> split_slabs;
				for (auto &slab : slabs)
				{
					split_slabs.push_back(std::make_shared<Slab>(slab->time_fe.get_degree(), slab->start_time, 0.5 * (slab->start_time + slab->end_time)));
					split_slabs.push_back(std::make_shared<Slab>(slab->time_fe.get_degree(), 0.5 * (slab->start_time + slab->end_time), slab->end_time));
				}
				slabs = split_slabs;

				for (auto &slab : slabs)
					GridGenerator::hyper_rectangle(slab->time_triangulation, Point<1>(slab->start_time), Point<1>(slab->end_time));
			}
			else
			{
				for (auto &slab : slabs)
					slab->time_triangulation.refine_global(refine_time);
			}
		}
	}

	print_convergence_table();
}

int main()
{
	try
	{
		deallog.depth_console(2);

		const std::string problem_name = "rotating_circle_long_term"; //"rotating_circle"; //"two_sources";
		const int dim = 2;											  // 1;

		std::vector<unsigned int> r;
		std::vector<double> t = {0.};
		int time_cycle = 5; // 0; // start from 0
		int time_factor = 10;
		double T = 10. * time_factor;
		int N = 16 * 4 * std::pow(2, time_cycle) * time_factor;
		double dt = T / N;
		for (unsigned int i = 0; i < N; ++i)
		{
			r.push_back(1);
			t.push_back((i + 1) * dt);
		}

		// run the simulation
		SpaceTime<dim> space_time_problem(
			problem_name,		 // problem_name
			1,					 // s ->  spatial FE degree
			r,					 // r -> temporal FE degree
			t,					 // time points
			(dim == 2) ? 6 : 10, // max_n_refinement_cycles,
			true,				 // refine_space
			false,				 // refine_time
			false				 // split_slabs
		);
		space_time_problem.run();

		// save final grid
		space_time_problem.print_grids("space_grid.svg", "time_grid.svg");
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl
				  << std::endl
				  << "----------------------------------------------------"
				  << std::endl;
		std::cerr << "Exception on processing: " << std::endl
				  << exc.what()
				  << std::endl
				  << "Aborting!" << std::endl
				  << "----------------------------------------------------"
				  << std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl
				  << std::endl
				  << "----------------------------------------------------"
				  << std::endl;
		std::cerr << "Unknown exception!" << std::endl
				  << "Aborting!"
				  << std::endl
				  << "----------------------------------------------------"
				  << std::endl;
		return 1;
	}

	return 0;
}
