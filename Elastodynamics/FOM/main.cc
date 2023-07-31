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
 * Tensor-product space-time code for the elastodynamics equation with Q^s finite elements in space and Q^r finite elements in time: cG(s)cG(r)
 * The time interval I = (0,T) is being divided into subintervals I_1, I_2, ..., I_M
 * and then instead of solving the PDE all at once on the full space-time domain Q := I x Ω
 * we solve sequentially (forward-in-time) on time slabs Q_n := I_n x Ω to reduce the size of the linear systems.
 * This also allows the usage of different temporal polynomial degrees on each slab, i.e. r := (r_1, r_2, ..., r_M) is now a multiindex.
 *
 * Author: Hendrik Fischer and Julian Roth, 2022-2023
 *
 * For more information on tensor-product space-time finite elements please also check out the DTM-project by Uwe Köcher and contributors.
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

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
class InitialValues : public Function<dim>
{
public:
	InitialValues() : Function<dim>(2) {}

	virtual double value(const Point<dim> &p,
						 const unsigned int component) const override;

	virtual void vector_value(const Point<dim> &p,
							  Vector<double> &value) const;
};

template <int dim>
double InitialValues<dim>::value(const Point<dim> &p,
								 const unsigned int component) const
{
	if (component == 0)
		return 0.;
	else if (component == 1)
		return p[0] * (1. - p[0]);
}

template <int dim>
void InitialValues<dim>::vector_value(const Point<dim> &p,
									  Vector<double> &values) const
{
	for (unsigned int c = 0; c < 2; ++c)
		values(c) = InitialValues<dim>::value(p, c);
}

template <int dim>
class InitialValues2D : public Function<dim>
{
public:
	InitialValues2D() : Function<dim>(2) {}

	virtual double value(const Point<dim> &p,
						 const unsigned int component) const override;

	virtual void vector_value(const Point<dim> &p,
							  Vector<double> &value) const;
};

template <int dim>
double InitialValues2D<dim>::value(const Point<dim> &p,
								   const unsigned int component) const
{
	double x_s = p[0] / 0.01;
	double y_s = p[1] / 0.01;
	if (component == 0)
	{
		if (1. - std::sqrt(x_s * x_s + y_s * y_s) < 0.)
			return 0.;
		else
			return std::exp(-x_s * x_s - y_s * y_s) * (1. - x_s * x_s - y_s * y_s);
	}
	else if (component == 1)
		return 0.;
}

template <int dim>
void InitialValues2D<dim>::vector_value(const Point<dim> &p,
										Vector<double> &values) const
{
	for (unsigned int c = 0; c < 2; ++c)
		values(c) = InitialValues2D<dim>::value(p, c);
}

template <int dim>
class Solution : public Function<dim>
{
public:
	Solution() : Function<dim>(2 * dim) {}

	virtual double value(const Point<dim> &p,
						 const unsigned int component = 0) const override;

	virtual void vector_value(const Point<dim> &p,
							  Vector<double> &value) const;
};

template <int dim>
double Solution<dim>::value(const Point<dim> &p,
							const unsigned int component) const
{
	const double t = this->get_time();
	switch (dim)
	{
	case 1:
	{
		if (component == 0)
			return std::sin(t) * p[0] * (1. - p[0]); // u(t,x) = sin(t)x(1-x)
		else if (component == 1)
			return std::cos(t) * p[0] * (1. - p[0]); // v(t,x) = cos(t)x(1-x)
	}
	case 2:
	{
		return 0.;
	}
	case 3:
	{
		return 0.;
	}
	default:
	{
		Assert(false, ExcNotImplemented());
	}
	}
	return -1.0; // to avoid "no return warning"
}

template <int dim>
void Solution<dim>::vector_value(const Point<dim> &p,
								 Vector<double> &values) const
{
	for (unsigned int c = 0; c < 2 * dim; ++c)
		values(c) = Solution<dim>::value(p, c);
}

template <int dim>
class RightHandSide : public Function<dim>
{
public:
	RightHandSide() : Function<dim>(dim)
	{
	}
	virtual double value(const Point<dim> &p,
						 const unsigned int component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
								 const unsigned int component) const
{
	const double t = this->get_time();
	// return 1.;

	switch (dim)
	{
	case 1:
		return (2. - p[0] * (1. - p[0])) * std::sin(t);
	case 2:
		return 0.;
	case 3:
	{
		double max_force = 0.5;
		double time_of_max_force = 5.;
		double time_2_release_force = 1.;
		// bool force_on = (t <= 10.);
		if (component == 0)
			return 0.;
		else if (component == 1)
			if (t <= time_of_max_force)
				return max_force * (t / time_of_max_force);
			else if (t <= time_of_max_force + time_2_release_force)
				return max_force * (1 - (t - time_of_max_force) / time_2_release_force);
			else
				return 0.;
		else if (component == 2)
			return 0.;
	}
	default:
	{
		Assert(false, ExcNotImplemented());
	}
	}
}

class Slab
{
public:
	// constructor
	Slab(unsigned int r, double start_time, double end_time);

	// variables
	Triangulation<1> time_triangulation;
	FE_Q<1> time_fe;
	DoFHandler<1> time_dof_handler;

	double start_time, end_time;
};

Slab::Slab(unsigned int r, double start_time, double end_time) : time_fe(r), time_dof_handler(time_triangulation), start_time(start_time), end_time(end_time)
{
}

template <int dim>
class SpaceTime
{
public:
	SpaceTime(std::string problem_name, int s, std::vector<unsigned int> r, std::vector<double> time_points = {0., 1.},
			  unsigned int max_n_refinement_cycles = 3, bool refine_space = true, bool refine_time = true, bool split_slabs = true);
	void run();
	void print_grids(std::string file_name_space, std::string file_name_time);
	void print_convergence_table();

private:
	void make_grids();
	void setup_system(std::shared_ptr<Slab> &slab, unsigned int k);
	void assemble_system(std::shared_ptr<Slab> &slab, unsigned int slab_number, unsigned int cycle, bool first_slab, bool assemble_matrix);
	void apply_initial_condition();
	void apply_boundary_conditions(std::shared_ptr<Slab> &slab, unsigned int cycle, bool first_slab);
	void solve(bool invert);
	void output_results(std::shared_ptr<Slab> &slab, const unsigned int refinement_cycle, unsigned int slab_number);
	void compute_functional_values(const unsigned int refinement_cycle, unsigned int slab_number);
	Tensor<1, dim> compute_normal_stress(Vector<double> space_solution);
	void process_solution(std::shared_ptr<Slab> &slab, const unsigned int cycle, bool last_slab);
	void print_coordinates(std::shared_ptr<Slab> &slab, std::string output_dir, unsigned int slab_number);

	std::string problem_name;

	// space
	Triangulation<dim> space_triangulation;
	FESystem<dim> space_fe;
	DoFHandler<dim> space_dof_handler;

	// time
	std::vector<std::shared_ptr<Slab>> slabs;

	// space-time
	SparsityPattern sparsity_pattern;
	SparseMatrix<double> system_matrix;
	SparseMatrix<double> mass_matrix;
	SparseDirectUMFPACK A_direct;
	Vector<double> solution;
	Vector<double> initial_solution; // u(0) or u(t_0)
	Vector<double> system_rhs;
	Vector<double> dual_rhs;

	FEValuesExtractors::Vector displacement;
	FEValuesExtractors::Vector velocity;
	types::global_dof_index n_space_u;
	types::global_dof_index n_space_v;

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

	double E = 1000.0;
	double nu = 0.3;
	double mu = E / (2. * (1. + nu));
	double lambda = E * nu / ((1. + nu) * (1. - 2. * nu));
};

template <int dim>
SpaceTime<dim>::SpaceTime(std::string problem_name, int s, std::vector<unsigned int> r, std::vector<double> time_points,
						  unsigned int max_n_refinement_cycles, bool refine_space, bool refine_time,
						  bool split_slabs) : problem_name(problem_name),
											  space_fe(/*u*/ FE_Q<dim>(s), dim, /*v*/ FE_Q<dim>(s), dim),
											  space_dof_handler(space_triangulation),
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
void SpaceTime<3>::print_grids(std::string file_name_space, std::string file_name_time)
{
	// TODO: Implement grid output in 3+1D, but migt also be a bit too expensive
	//	// space
	//	std::ofstream out_space(file_name_space);
	//	GridOut grid_out_space;
	//	grid_out_space.write_svg(space_triangulation, out_space);
	//
	//	// time
	//	std::ofstream out_time(file_name_time);
	//	print_1d_grid_slabwise(out_time, slabs, start_time, end_time);
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
	space_triangulation.refine_global(1);
	for (auto &slab : slabs)
		slab->time_triangulation.refine_global(0);
}

template <>
void SpaceTime<2>::make_grids()
{
	// Simplified version of Example 5.4 from Bangerth, Geiger, Rannacher paper

	// create grids
	GridGenerator::hyper_rectangle(space_triangulation, Point<2>(-5., -5.), Point<2>(5., 5.));
	for (auto &slab : slabs)
		GridGenerator::hyper_rectangle(slab->time_triangulation, Point<1>(slab->start_time), Point<1>(slab->end_time));

	// only Neumann BC
	for (auto &cell : space_triangulation.cell_iterators())
		for (unsigned int face = 0; face < GeometryInfo<2>::faces_per_cell; face++)
			if (cell->face(face)->at_boundary())
				cell->face(face)->set_boundary_id(1);

	// globally refine the grids
	space_triangulation.refine_global(1);
	for (auto &slab : slabs)
		slab->time_triangulation.refine_global(0);
}

template <>
void SpaceTime<3>::make_grids()
{
	// Example from https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/elastodynamics/demo_elastodynamics.py.html

	// create grids
	GridGenerator::subdivided_hyper_rectangle(space_triangulation, {6, 1, 1}, Point<3>(0., 0., 0.), Point<3>(6., 1., 1.));
	for (auto &slab : slabs)
		GridGenerator::hyper_rectangle(slab->time_triangulation, Point<1>(slab->start_time), Point<1>(slab->end_time));

	for (auto &cell : space_triangulation.cell_iterators())
		for (unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell; face++)
			if (cell->face(face)->at_boundary())
			{
				if (cell->face(face)->center()[0] < 1e-10)
					cell->face(face)->set_boundary_id(0); //   hom. Dirichlet BC
				else if (cell->face(face)->center()[1] > (1. - 1.e-10))
					cell->face(face)->set_boundary_id(2); // inhom. Neumann BC
				else
					cell->face(face)->set_boundary_id(1); //   hom. Neumann BC
			}

	// globally refine the grids
	space_triangulation.refine_global(0);
	for (auto &slab : slabs)
		slab->time_triangulation.refine_global(0);
}

template <int dim>
void SpaceTime<dim>::setup_system(std::shared_ptr<Slab> &slab, unsigned int k)
{
	std::cout << "Slab Q_" << k << " = Ω x (" << slab->start_time << "," << slab->end_time << "):" << std::endl;
	std::cout << "   Finite Elements: cG(" << space_fe.degree << ") (space), cG("
			  << slab->time_fe.get_degree() << ") (time)" << std::endl;

	slab->time_dof_handler.distribute_dofs(slab->time_fe);

	std::cout << "   Number of active cells: " << space_triangulation.n_active_cells() << " (space), "
			  << slab->time_triangulation.n_active_cells() << " (time)" << std::endl;
	std::cout << "   Number of degrees of freedom: " << space_dof_handler.n_dofs()
			  << " (" << n_space_u << '+' << n_space_v << ')' << " (space), "
			  << slab->time_dof_handler.n_dofs() << " (time)" << std::endl;

	/////////////////////////////////////////////////////////////////////////////////////////
	// space-time sparsity pattern = tensor product of spatial and temporal sparsity pattern
	//

	// renumber temporal DoFs, which for cG(2) are numbered 0, 2, 1 and not 0, 1, 2
	// time
	std::vector<Point<1>> _time_support_points(slab->time_dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(
		MappingQ1<1, 1>(),
		slab->time_dof_handler,
		_time_support_points); // {0., 10., 5.}

	std::map<double, types::global_dof_index, std::less<double>> time_support_map;
	for (unsigned int i = 0; i < _time_support_points.size(); ++i)
		time_support_map[_time_support_points[i][0]] = i;

	// sort time_support_points by time
	std::sort(std::begin(_time_support_points), std::end(_time_support_points), [](Point<1> const &a, Point<1> const &b)
			  { return a[0] < b[0]; });

	std::vector<types::global_dof_index> time_support_points_order_new(_time_support_points.size());
	for (unsigned int i = 0; i < _time_support_points.size(); ++i)
		time_support_points_order_new[time_support_map[_time_support_points[i][0]]] = i;

	slab->time_dof_handler.renumber_dofs(time_support_points_order_new);

	std::vector<Point<1>> time_support_points_new(slab->time_dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(
		MappingQ1<1, 1>(),
		slab->time_dof_handler,
		time_support_points_new); // {0., 5., 10.}

	// for debugging:
	// for (auto& t : time_support_points_new)
	// 	std::cout << "t = " << t[0] << std::endl;

	if (k == 1)
	{
		// spatial sparsity pattern
		DynamicSparsityPattern space_dsp(space_dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(space_dof_handler, space_dsp);

		SparsityPattern space_sparsity_pattern;
		space_sparsity_pattern.copy_from(space_dsp);
		std::ofstream out_space_sparsity("space_sparsity_pattern.svg");
		space_sparsity_pattern.print_svg(out_space_sparsity);

		// temporal sparsity pattern
		DynamicSparsityPattern time_dsp(slab->time_dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(slab->time_dof_handler, time_dsp);

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
	}

	solution.reinit(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
	system_rhs.reinit(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
	dual_rhs.reinit(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
}

template <int dim>
void SpaceTime<dim>::assemble_system(std::shared_ptr<Slab> &slab, unsigned int slab_number, unsigned int cycle, bool first_slab, bool assemble_matrix)
{
	RightHandSide<dim> right_hand_side;

	// space
	QGauss<dim> space_quad_formula(space_fe.degree + 2);
	QGauss<dim - 1> space_face_quad_formula(space_fe.degree + 2);
	FEValues<dim> space_fe_values(space_fe, space_quad_formula,
								  update_values | update_gradients | update_quadrature_points | update_JxW_values);
	FEFaceValues<dim> space_fe_face_values(space_fe, space_face_quad_formula,
										   update_values | update_gradients | update_normal_vectors | update_JxW_values);
	const unsigned int space_dofs_per_cell = space_fe.n_dofs_per_cell();
	std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);

	// time
	QGauss<1> time_quad_formula(slab->time_fe.degree + 2);
	FEValues<1> time_fe_values(slab->time_fe, time_quad_formula,
							   update_values | update_gradients | update_quadrature_points | update_JxW_values);
	const unsigned int time_dofs_per_cell = slab->time_fe.n_dofs_per_cell();
	std::vector<types::global_dof_index> time_local_dof_indices(time_dofs_per_cell);

	// local contributions on space-time cell
	FullMatrix<double> cell_matrix(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
	FullMatrix<double> cell_mass_matrix(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
	Vector<double> cell_rhs(space_dofs_per_cell * time_dofs_per_cell);
	Vector<double> cell_dual_rhs(space_dofs_per_cell * time_dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices(space_dofs_per_cell * time_dofs_per_cell);

	Tensor<2, dim> identity;
	for (unsigned int k = 0; k < dim; ++k)
		identity[k][k] = 1.;

	// locally assemble on each space-time cell
	for (const auto &space_cell : space_dof_handler.active_cell_iterators())
	{
		space_fe_values.reinit(space_cell);
		space_cell->get_dof_indices(space_local_dof_indices);

		if (assemble_matrix)
		{
			for (const auto &time_cell : slab->time_dof_handler.active_cell_iterators())
			{
				time_fe_values.reinit(time_cell);
				time_cell->get_dof_indices(time_local_dof_indices);

				cell_matrix = 0;
				cell_mass_matrix = 0;
				cell_dual_rhs = 0;

				for (const unsigned int qq : time_fe_values.quadrature_point_indices())
				{
					// time quadrature point
					const double t_qq = time_fe_values.quadrature_point(qq)[0];
					right_hand_side.set_time(t_qq);

					for (const unsigned int q : space_fe_values.quadrature_point_indices())
					{
						// space quadrature point
						const auto x_q = space_fe_values.quadrature_point(q);

						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
							{
								const Tensor<2, dim> symmetric_gradient_i = 0.5 * (space_fe_values[displacement].gradient(i, q) + transpose(space_fe_values[displacement].gradient(i, q)));
								const Tensor<2, dim> stress_tensor_i = 2. * mu * symmetric_gradient_i + lambda * trace(symmetric_gradient_i) * identity;

								// system matrix
								for (const unsigned int j : space_fe_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
									{
										const Tensor<2, dim> symmetric_gradient_j = 0.5 * (space_fe_values[displacement].gradient(j, q) + transpose(space_fe_values[displacement].gradient(j, q)));

										cell_matrix(
											j + jj * space_dofs_per_cell,
											i + ii * space_dofs_per_cell) += (space_fe_values[velocity].value(i, q) * time_fe_values.shape_grad(ii, qq)[0] *	 // ∂_t ϕ^v_{i,ii}(t_qq, x_q)
																				  space_fe_values[displacement].value(j, q) * time_fe_values.shape_value(jj, qq) //     ϕ^u_{j,jj}(t_qq, x_q)
																																								 // +
																			  + scalar_product(
																					stress_tensor_i * time_fe_values.shape_value(ii, qq),	  //  σ(ϕ^u)_{i,ii}(t_qq, x_q)
																					symmetric_gradient_j * time_fe_values.shape_value(jj, qq) // ∇_x ϕ^u_{j,jj}(t_qq, x_q)
																					)
																			  // +
																			  + space_fe_values[displacement].value(i, q) * time_fe_values.shape_grad(ii, qq)[0] * // ∂_t ϕ^u_{i,ii}(t_qq, x_q)
																					space_fe_values[velocity].value(j, q) * time_fe_values.shape_value(jj, qq)	   //     ϕ^v_{j,jj}(t_qq, x_q)
																																								   // -
																			  - space_fe_values[velocity].value(i, q) * time_fe_values.shape_value(ii, qq) *	   //  ϕ^v_{i,ii}(t_qq, x_q)
																					space_fe_values[velocity].value(j, q) * time_fe_values.shape_value(jj, qq)	   //  ϕ^v_{j,jj}(t_qq, x_q)
																			  ) *
																			 space_fe_values.JxW(q) * time_fe_values.JxW(qq); // d(t,x)
									}
							}
					}
				}

				// assemble mass matrix for initial condition; still enforcing initial condition, but need this mass term for the dual problem
				if (time_cell->active_cell_index() == 0)
				{
					for (const unsigned int q : space_fe_values.quadrature_point_indices())
						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int j : space_fe_values.dof_indices())
								cell_mass_matrix(
									j + 0 * space_dofs_per_cell,
									i + 0 * space_dofs_per_cell) += (space_fe_values[displacement].value(i, q) * //  ϕ^u_{i,0}(0, x_q)
																		 space_fe_values[velocity].value(j, q)	 //  ϕ^v_{j,0}(0, x_q)
																	 // +
																	 + space_fe_values[velocity].value(i, q) *		 //  ϕ^v_{i,0}(0, x_q)
																		   space_fe_values[displacement].value(j, q) //  ϕ^u_{j,0}(0, x_q)
																	 ) *
																	space_fe_values.JxW(q); //  d(x)
				}

				// distribute local to global
				if (assemble_matrix)
					for (const unsigned int i : space_fe_values.dof_indices())
						for (const unsigned int ii : time_fe_values.dof_indices())
						{
							// system matrix
							for (const unsigned int j : space_fe_values.dof_indices())
								for (const unsigned int jj : time_fe_values.dof_indices())
								{
									system_matrix.add(
										space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs(),
										space_local_dof_indices[j] + time_local_dof_indices[jj] * space_dof_handler.n_dofs(),
										cell_matrix(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));

									mass_matrix.add(
										space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs(),
										space_local_dof_indices[j] + time_local_dof_indices[jj] * space_dof_handler.n_dofs(),
										cell_mass_matrix(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));
								}
						}
			}
		}

		for (const unsigned int space_face : space_cell->face_indices())
		{
			if (space_cell->at_boundary(space_face) && (space_cell->face(space_face)->boundary_id() == 2)) // face is at inhom. Neumann boundary
			{
				space_fe_face_values.reinit(space_cell, space_face);
				for (const auto &time_cell : slab->time_dof_handler.active_cell_iterators())
				{
					time_fe_values.reinit(time_cell);
					time_cell->get_dof_indices(time_local_dof_indices);

					cell_rhs = 0;

					for (const unsigned int qq : time_fe_values.quadrature_point_indices())
					{
						// time quadrature point
						const double t_qq = time_fe_values.quadrature_point(qq)[0];
						right_hand_side.set_time(t_qq);

						for (const unsigned int q : space_fe_face_values.quadrature_point_indices())
						{
							// space quadrature point
							const auto x_q = space_fe_values.quadrature_point(q);
							// precompute RHS value at (t_qq, x_q)
							Tensor<1, dim> rhs_tensor_value;
							for (unsigned int k = 0; k < dim; k++)
								rhs_tensor_value[k] = right_hand_side.value(x_q, k);

							for (const unsigned int i : space_fe_face_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
								{
									auto tmp = (space_fe_face_values[displacement].value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ^v_{i,ii}(t_qq, x_q)
												rhs_tensor_value																	  //           g(t_qq, x_q)
												) *
											   space_fe_face_values.JxW(q) * time_fe_values.JxW(qq); // d(t,x)

									cell_rhs(
										i + ii * space_dofs_per_cell) += tmp;
								}
						}
					}

					// distribute local to global
					for (const unsigned int i : space_fe_face_values.dof_indices())
						for (const unsigned int ii : time_fe_values.dof_indices())
							system_rhs(space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs()) += cell_rhs(i + ii * space_dofs_per_cell);
				}
			}
			else if (space_cell->at_boundary(space_face) && (space_cell->face(space_face)->boundary_id() == 0)) // face is at hom. Dirichlet boundary
			{
				space_fe_face_values.reinit(space_cell, space_face);
				for (const auto &time_cell : slab->time_dof_handler.active_cell_iterators())
				{
					time_fe_values.reinit(time_cell);
					time_cell->get_dof_indices(time_local_dof_indices);

					cell_dual_rhs = 0;

					for (const unsigned int qq : time_fe_values.quadrature_point_indices())
					{
						for (const unsigned int q : space_fe_face_values.quadrature_point_indices())
						{
							for (const unsigned int i : space_fe_face_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
								{
									const Tensor<2, dim> symmetric_gradient_i = 0.5 * (space_fe_face_values[displacement].gradient(i, q) + transpose(space_fe_face_values[displacement].gradient(i, q)));
									const Tensor<2, dim> stress_tensor_i = 2. * mu * symmetric_gradient_i + lambda * trace(symmetric_gradient_i) * identity;

									cell_dual_rhs(
										i + ii * space_dofs_per_cell) += (1. / (end_time - start_time)) * (stress_tensor_i * space_fe_face_values.normal_vector(q))[1] * space_fe_face_values.JxW(q) * time_fe_values.JxW(qq); // J = σ(ϕ^u)_{i,ii}(t_qq, x_q) · n(x_q) · (1,0,0) d(t,x)
								}
						}
					}

					// distribute local to global
					for (const unsigned int i : space_fe_face_values.dof_indices())
						for (const unsigned int ii : time_fe_values.dof_indices())
							dual_rhs(space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs()) += cell_dual_rhs(i + ii * space_dofs_per_cell);
				}
			}
		}
	}

	std::string output_dir = "../Data/" + std::to_string(dim) + "D/" + problem_name + "/cycle=" + std::to_string(cycle) + "-0/"; // HF: set time cycle

	/////////////////////////////////////////////
	// save system matrix and rhs to file (NO BC)
	if (first_slab)
	{
		std::cout << output_dir + "matrix_no_bc.txt" << std::endl;
		std::ofstream matrix_no_bc_out(output_dir + "matrix_no_bc.txt");
		print_as_numpy_arrays_high_resolution(system_matrix, matrix_no_bc_out, /*precision*/ 16);

		std::ofstream mass_matrix_no_bc_out(output_dir + "mass_matrix_no_bc.txt");
		print_as_numpy_arrays_high_resolution(mass_matrix, mass_matrix_no_bc_out, /*precision*/ 16);
	}

	std::ofstream rhs_no_bc_out(output_dir + "primal_rhs_no_bc_" + Utilities::int_to_string(slab_number, 5) + ".txt");
	system_rhs.print(rhs_no_bc_out, /*precision*/ 16);

	std::ofstream dual_rhs_no_bc_out(output_dir + "dual_rhs_no_bc_" + Utilities::int_to_string(slab_number, 5) + ".txt");
	dual_rhs.print(dual_rhs_no_bc_out, /*precision*/ 16);

	//	std::cout << "BEFORE IC - system_rhs: " << system_rhs.l2_norm() << std::endl;
	apply_initial_condition();
	//	std::cout << "AFTER IC - system_rhs: " << system_rhs.l2_norm() << std::endl;
	apply_boundary_conditions(slab, cycle, first_slab);
	//	std::cout << "AFTER BC - system_rhs: " << system_rhs.l2_norm() << std::endl;
}

template <int dim>
void SpaceTime<dim>::apply_initial_condition()
{
	// apply initial value on linear system

	// clear first N_space rows of system matrix
	for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
	{
		// iterate over entries of i.th row of system matrix
		// A_ij = δ_ij
		for (typename SparseMatrix<double>::iterator p = system_matrix.begin(i); p != system_matrix.end(i); ++p)
			p->value() = 0.;
		system_matrix.set(i, i, 1.);
		system_rhs(i) = initial_solution(i);
	}
}

template <int dim>
void SpaceTime<dim>::apply_boundary_conditions(std::shared_ptr<Slab> &slab, unsigned int cycle, bool first_slab)
{
	// apply the spatial Dirichlet boundary conditions at each temporal DoF
	Solution<dim> solution_func;

	FEValues<1> time_fe_values(slab->time_fe, Quadrature<1>(slab->time_fe.get_unit_support_points()), update_quadrature_points);
	std::vector<types::global_dof_index> time_local_dof_indices(slab->time_fe.n_dofs_per_cell());

	// list of constrained space-time DoF indices
	std::vector<types::global_dof_index> boundary_ids;

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
			VectorTools::interpolate_boundary_values(space_dof_handler, 0, solution_func /*ZeroFunction<dim>(1+1)*/, boundary_values);

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
		std::ofstream boundary_id_out("../Data/3D/" + problem_name + "/cycle=" + std::to_string(cycle) + "-0/boundary_id.txt"); // HF: set time cycle
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

template <>
void SpaceTime<1>::output_results(std::shared_ptr<Slab> &slab, const unsigned int refinement_cycle, unsigned int slab_number)
{
	std::string output_dir = "../Data/1D/" + problem_name + "/cycle=" + std::to_string(refinement_cycle) + "-0/"; // HF: set time cycle

	//////////////////////////////////////////////////////////////////////////////
	// output space-time vectors for displacement and velocity separately as .txt
	//
	Vector<double> solution_u(n_space_u * slab->time_dof_handler.n_dofs());
	Vector<double> solution_v(n_space_v * slab->time_dof_handler.n_dofs());
	unsigned int index_u = 0;
	unsigned int index_v = 0;
	for (unsigned int ii = 0; ii < slab->time_dof_handler.n_dofs(); ++ii)
	{
		// displacement
		for (unsigned int j = ii * space_dof_handler.n_dofs(); j < ii * space_dof_handler.n_dofs() + n_space_u; ++j)
		{
			solution_u[index_u] = solution[j];
			index_u++;
		}
		// velocity
		for (unsigned int j = ii * space_dof_handler.n_dofs() + n_space_u; j < (ii + 1) * space_dof_handler.n_dofs(); ++j)
		{
			solution_v[index_v] = solution[j];
			index_v++;
		}
	}
	std::ofstream solution_u_out(output_dir + "solution_u_" + Utilities::int_to_string(slab_number, 5) + ".txt");
	solution_u.print(solution_u_out, /*precision*/ 16);
	std::ofstream solution_v_out(output_dir + "solution_v_" + Utilities::int_to_string(slab_number, 5) + ".txt");
	solution_v.print(solution_v_out, /*precision*/ 16);

	///////////////////////////////////////////
	// output coordinates for t and x as .txt
	//

	// space
	std::vector<Point<1>> space_support_points(space_dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(
		MappingQ1<1, 1>(),
		space_dof_handler,
		space_support_points);

	std::ofstream x_out(output_dir + "coordinates_x.txt");
	x_out.precision(9);
	x_out.setf(std::ios::scientific, std::ios::floatfield);

	unsigned int i = 0;
	for (auto point : space_support_points)
	{
		if (i == n_space_u)
			break;
		x_out << point[0] << ' ';
		i++;
	}
	x_out << std::endl;

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

template <>
void SpaceTime<2>::output_results(std::shared_ptr<Slab> &slab, const unsigned int refinement_cycle, unsigned int slab_number)
{
	std::string output_dir = "../Data/2D/" + problem_name + "/cycle=" + std::to_string(refinement_cycle) + "-0/"; // HF: set time cycle

	if (slab_number == 0)
	{
		std::ofstream dof_output(output_dir + "dof.txt");

		FEValues<2> space_fe_values(space_fe, Quadrature<2>(space_fe.get_unit_support_points()),
									update_values | update_gradients | update_quadrature_points | update_JxW_values);
		const unsigned int space_dofs_per_cell = space_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);

		for (const auto &space_cell : space_dof_handler.active_cell_iterators())
		{
			space_fe_values.reinit(space_cell);
			space_cell->get_dof_indices(space_local_dof_indices);

			for (const unsigned int q : space_fe_values.quadrature_point_indices())
			{
				// space quadrature point
				const auto x_q = space_fe_values.quadrature_point(q);
				if (space_local_dof_indices[q] < n_space_u)
					dof_output << space_local_dof_indices[q] << " ";
				// std::cout << x_q[0] << " " << x_q[1] << " " << space_local_dof_indices[q] << std::endl;
			}
		}
		dof_output << std::endl;
	}

	// output results as VTK files
	unsigned int ii_ordered = 0;
	for (auto time_point : time_support_points)
	{
		if (slab_number > 0 && ii_ordered == 0)
		{
			ii_ordered++;
			continue;
		}

		double t_qq = time_point.first;
		unsigned int ii = time_point.second;

		DataOut<2> data_out;
		data_out.attach_dof_handler(space_dof_handler);

		Vector<double> space_solution(space_dof_handler.n_dofs());
		for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
			space_solution(i) = solution(i + ii * space_dof_handler.n_dofs());

		std::vector<std::string> solution_names;
		solution_names.push_back("x_disp");
		solution_names.push_back("y_disp");
		solution_names.push_back("x_velo");
		solution_names.push_back("y_velo");

		std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(2 + 2, DataComponentInterpretation::component_is_scalar);
		data_out.add_data_vector(space_solution, solution_names, DataOut<2>::type_dof_data, data_component_interpretation);
		data_out.build_patches(1);

		data_out.set_flags(DataOutBase::VtkFlags(t_qq, ii));

		std::ofstream output(output_dir + "solution" + Utilities::int_to_string(n_snapshots, 5) + ".vtk");
		data_out.write_vtk(output);

		++ii_ordered;
		++n_snapshots;
	}
}

template <>
void SpaceTime<3>::output_results(std::shared_ptr<Slab> &slab, const unsigned int refinement_cycle, unsigned int slab_number)
{
	std::string output_dir = "../Data/3D/" + problem_name + "/cycle=" + std::to_string(refinement_cycle) + "-0/"; // HF: set time cycle

	if (slab_number == 0)
	{
		std::ofstream dof_output(output_dir + "dof.txt");

		FEValues<3> space_fe_values(space_fe, Quadrature<3>(space_fe.get_unit_support_points()),
									update_values | update_gradients | update_quadrature_points | update_JxW_values);
		const unsigned int space_dofs_per_cell = space_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);

		for (const auto &space_cell : space_dof_handler.active_cell_iterators())
		{
			space_fe_values.reinit(space_cell);
			space_cell->get_dof_indices(space_local_dof_indices);

			for (const unsigned int q : space_fe_values.quadrature_point_indices())
			{
				// space quadrature point
				const auto x_q = space_fe_values.quadrature_point(q);
				if (space_local_dof_indices[q] < n_space_u)
					dof_output << space_local_dof_indices[q] << " ";
				// std::cout << x_q[0] << " " << x_q[1] << " " << space_local_dof_indices[q] << std::endl;
			}
		}
		dof_output << std::endl;
	}

	// output results as VTK files
	unsigned int ii_ordered = 0;
	for (auto time_point : time_support_points)
	{
		if (slab_number > 0 && ii_ordered == 0)
		{
			ii_ordered++;
			continue;
		}

		double t_qq = time_point.first;
		unsigned int ii = time_point.second;

		//   std::cout << "n_snapshots = " << n_snapshots << ", t_qq = " << t_qq << ", ii = " << ii << std::endl;

		DataOut<3> data_out;
		data_out.attach_dof_handler(space_dof_handler);

		Vector<double> space_solution(space_dof_handler.n_dofs());
		for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
			space_solution(i) = solution(i + ii * space_dof_handler.n_dofs());

		std::vector<std::string> solution_names;
		solution_names.push_back("x_disp");
		solution_names.push_back("y_disp");
		solution_names.push_back("z_disp");
		solution_names.push_back("x_velo");
		solution_names.push_back("y_velo");
		solution_names.push_back("z_velo");

		std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(3 + 3, DataComponentInterpretation::component_is_scalar);
		data_out.add_data_vector(space_solution, solution_names, DataOut<3>::type_dof_data, data_component_interpretation);
		data_out.build_patches(1);

		data_out.set_flags(DataOutBase::VtkFlags(t_qq, ii));

		std::ofstream output(output_dir + "solution" + Utilities::int_to_string(n_snapshots, 5) + ".vtk");
		data_out.write_vtk(output);

		++ii_ordered;
		++n_snapshots;
	}
}

template <int dim>
void SpaceTime<dim>::compute_functional_values(const unsigned int refinement_cycle, unsigned int slab_number)
{
	std::string output_dir = "../Data/3D/" + problem_name + "/cycle=" + std::to_string(refinement_cycle) + "-0/"; // HF: set time cycle

	std::vector<std::tuple<double, double>> stress_x_values;
	std::vector<std::tuple<double, double>> stress_y_values;
	std::vector<std::tuple<double, double>> stress_z_values;

	unsigned int ii_ordered = 0;
	for (auto time_point : time_support_points)
	{
		if (slab_number > 0 && ii_ordered == 0)
		{
			ii_ordered++;
			continue;
		}

		double t_qq = time_point.first;
		unsigned int ii = time_point.second;

		DataOut<3> data_out;
		data_out.attach_dof_handler(space_dof_handler);

		Vector<double> space_solution(space_dof_handler.n_dofs());
		for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
			space_solution(i) = solution(i + ii * space_dof_handler.n_dofs());

		Tensor<1, dim> normal_stress = compute_normal_stress(space_solution);
		stress_x_values.push_back(std::make_tuple(t_qq, normal_stress[0]));
		stress_y_values.push_back(std::make_tuple(t_qq, normal_stress[1]));
		stress_z_values.push_back(std::make_tuple(t_qq, normal_stress[2]));
	}

	// sort value vectors according to time
	std::sort(stress_x_values.begin(), stress_x_values.end());
	std::sort(stress_y_values.begin(), stress_y_values.end());
	std::sort(stress_z_values.begin(), stress_z_values.end());

	// output the values into log files
	std::ofstream stress_x_out(output_dir + "stress_x.txt", std::ios_base::app);
	std::ofstream stress_y_out(output_dir + "stress_y.txt", std::ios_base::app);
	std::ofstream stress_z_out(output_dir + "stress_z.txt", std::ios_base::app);

	for (auto &item : stress_x_values)
		stress_x_out << std::get<0>(item) << "," << std::get<1>(item) << std::endl;

	for (auto &item : stress_y_values)
		stress_y_out << std::get<0>(item) << "," << std::get<1>(item) << std::endl;

	for (auto &item : stress_z_values)
		stress_z_out << std::get<0>(item) << "," << std::get<1>(item) << std::endl;
}

template <int dim>
Tensor<1, dim> SpaceTime<dim>::compute_normal_stress(Vector<double> space_solution)
{
	Tensor<1, dim> normal_stress;

	QGauss<dim - 1> space_face_quad_formula(space_fe.degree + 2);
	FEFaceValues<dim> space_fe_face_values(space_fe, space_face_quad_formula,
										   update_values | update_gradients | update_normal_vectors | update_JxW_values | update_quadrature_points);

	const unsigned int space_dofs_per_cell = space_fe.n_dofs_per_cell();
	const unsigned int n_face_q_points = space_face_quad_formula.size();

	std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);
	std::vector<std::vector<Tensor<1, dim>>> face_solution_grads(n_face_q_points, std::vector<Tensor<1, dim>>(2 * dim));

	Tensor<2, dim> identity;
	for (unsigned int k = 0; k < dim; ++k)
		identity[k][k] = 1.;

	for (const auto &space_cell : space_dof_handler.active_cell_iterators())
	{
		for (const unsigned int space_face : space_cell->face_indices())
			if (space_cell->at_boundary(space_face) && (space_cell->face(space_face)->boundary_id() == 0)) // face is at hom. Dirichlet boundary
			{
				space_fe_face_values.reinit(space_cell, space_face);
				space_fe_face_values.get_function_gradients(space_solution, face_solution_grads);

				for (const unsigned int q : space_fe_face_values.quadrature_point_indices())
				{
					Tensor<2, dim> symm_grad_u;
					for (unsigned int l = 0; l < dim; l++)
						for (unsigned int m = 0; m < dim; m++)
							symm_grad_u[l][m] = 0.5 * (face_solution_grads[q][l][m] + face_solution_grads[q][m][l]);

					const Tensor<2, dim> stress_tensor = 2. * mu * symm_grad_u + lambda * trace(symm_grad_u) * identity;

					normal_stress += stress_tensor * space_fe_face_values.normal_vector(q) * space_fe_face_values.JxW(q);
				}
			}
	}

	return normal_stress;
}

template <int dim>
void SpaceTime<dim>::process_solution(std::shared_ptr<Slab> &slab, const unsigned int cycle, bool last_slab)
{
	Solution<dim> solution_func;

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

			// get the FEM space solution at the quadrature point
			Vector<double> space_solution(space_dof_handler.n_dofs());
			for (const unsigned int ii : time_fe_values.dof_indices())
			{
				for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
					space_solution(i) += solution(i + time_local_dof_indices[ii] * space_dof_handler.n_dofs()) * time_fe_values.shape_value(ii, qq);
			}

			// get the analytical space solution at the quadrature point
			Vector<double> analytical_solution(space_dof_handler.n_dofs());
			VectorTools::interpolate(space_dof_handler,
									 solution_func,
									 analytical_solution,
									 ComponentMask());

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
	n_time_dofs += (slab->time_dof_handler.n_dofs() - 1); // first time DoF is also part of last slab

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
		unsigned int i = 0;
		for (auto point : space_support_points)
		{
			if (i == n_space_u) // do not also output x-coordinates for the velocity, which are the same as for the displacement
				break;
			x_out << point[d] << ' ';
			i++;
		}
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

	displacement = FEValuesExtractors::Vector(0);
	velocity = FEValuesExtractors::Vector(dim);

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
		std::string dim_dir = "../Data/" + std::to_string(dim) + "D/";
		std::string problem_dir = dim_dir + problem_name + "/";
		std::string output_dir = problem_dir + "cycle=" + std::to_string(cycle) + "-0/"; // HF: set time cycle
		for (auto dir : {"../Data/", dim_dir.c_str(), problem_dir.c_str(), output_dir.c_str()})
			mkdir(dir, S_IRWXU);

		// remove old logfiles
		std::remove((output_dir + "stress_x.txt").c_str());
		std::remove((output_dir + "stress_y.txt").c_str());
		std::remove((output_dir + "stress_z.txt").c_str());

		// reset values from last refinement cycle
		n_snapshots = 0;
		n_active_time_cells = 0;
		n_time_dofs = 1;
		L2_error = 0.;

		////////////////////////////////////////////
		// initial value: u(0)
		//
		space_dof_handler.distribute_dofs(space_fe);

		// Renumber spatials DoFs into displacement and velocity DoFs
		// We are dealing with 2*dim components:
		// displacements [0 -- dim-1]:         0
		// velocities  [dim -- 2*dim-1]:       1
		std::vector<unsigned int> block_component(2 * dim, 0);
		for (unsigned int i = 0; i < dim; ++i)
			block_component[dim + i] = 1;
		DoFRenumbering::component_wise(space_dof_handler, block_component);

		// two blocks: displacement and velocity
		const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(space_dof_handler, block_component);
		n_space_u = dofs_per_block[0];
		n_space_v = dofs_per_block[1];

		// no hanging node constraints
		AffineConstraints<double> constraints;
		constraints.close();

		// compute initial value vector
		initial_solution.reinit(space_dof_handler.n_dofs());
		// TODO: if using non-zero initial values, create an initial values function of size 2*dim
		//		if (dim == 1)
		//		{
		//			VectorTools::interpolate(space_dof_handler,
		//				 InitialValues<dim>(),
		//				 initial_solution,
		//				 ComponentMask());
		//		}
		//		else if (dim == 2)
		//		{
		//			VectorTools::interpolate(space_dof_handler,
		//				 InitialValues2D<dim>(),
		//				 initial_solution,
		//				 ComponentMask());
		//		}
		std::ofstream initial_out(output_dir + "initial_solution.txt");
		initial_solution.print(initial_out, /*precision*/ 16);

		for (unsigned int k = 0; k < slabs.size(); ++k)
		{
			// create and solve linear system
			setup_system(slabs[k], k + 1);
			assemble_system(slabs[k], k, cycle, k == 0, k == 0);
			solve(k == 0);

			// output Space-Time DoF coordinates
			print_coordinates(slabs[k], output_dir, k);

			// output results as SVG or VTK files
			output_results(slabs[k], cycle, k);

			// compute normal stress at Dirichlet boundary
			compute_functional_values(cycle, k);

			// Compute the error to the analytical solution
			process_solution(slabs[k], cycle, (k == slabs.size() - 1));

			///////////////////////
			// prepare next slab
			//

			// get initial value for next slab
			auto last_time_point = std::prev(time_support_points.end());
			unsigned int ii = last_time_point->second;
			for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
				initial_solution(i) = solution(i + ii * space_dof_handler.n_dofs());

			// remove old temporal support points
			time_support_points.clear();

			std::cout << "Solution norm: " << solution.l2_norm() << std::endl;
			std::cout << "RHS norm: " << system_rhs.l2_norm() << std::endl;
		}

		if (dim == 1)
		{
			std::string plotting_cmd = "python3 plot_solution.py  " + output_dir;
			system(plotting_cmd.c_str());
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

		const std::string problem_name = "Rod";
		const int dim = 3;

		std::vector<unsigned int> r;
		std::vector<double> t = {0.};
		int time_cycle = 0; // start from 0
		double T = 40.;		//*2/4;
		// int N = 1600;	// 1600; //int(1600*2/4); //0; //2*4;
		int N = 200 * std::pow(2, time_cycle);
		double dt = T / N;
		for (unsigned int i = 0; i < N; ++i)
		{
			r.push_back(2);
			t.push_back((i + 1) * dt);
		}

		// run the simulation
		SpaceTime<dim> space_time_problem(
			problem_name, // problem_name
			1,			  // s ->  spatial FE degree
			r,			  // r -> temporal FE degree
			t,			  // time_points,
			4,			  // max_n_refinement_cycles,
			true,		  // refine_space
			false,		  // refine_time
			false		  // split_slabs
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
