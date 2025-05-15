package monte_carlo

import "core:math/rand"
import "core:fmt"
import "base:runtime"
import "core:thread"
import "core:sync"

polynomial_monte_carlo_single_thread :: proc (
    number_of_variables: i32,
    max_degrees: [^]i32,
    coefficients: [^]f32,
    region: [^]f32,
    n: i32,
) -> f32 {
    number_of_terms : i32 = 1
    for i in 0..<number_of_variables {
        number_of_terms *= max_degrees[i] + 1
    }

    for i in 0..<number_of_variables {
        assert(region[2 * i] < region[2 * i + 1])
    }

    point := make([dynamic]f32, number_of_variables)
    defer delete(point)
    term_degrees := make([dynamic]i32, number_of_variables)
    defer delete(term_degrees)

    res : f32 = 0.0
    for _ in 0..<n {
        for i in 0..<number_of_variables {
            point[i] = region[2*i] + rand.float32() * (region[2*i + 1] - region[2*i])
        }

        polynomial_value : f32 = 0.0
        for i in 0..<number_of_terms {
            term_value := coefficients[i]
            for j in 0..<number_of_variables {
                degree := term_degrees[j]
                for _ in 0..<degree {
                    term_value *= point[j]
                }
            }
            polynomial_value += term_value

            for j := number_of_variables - 1; j > -1; j -= 1 {
                if term_degrees[j] == max_degrees[j] {
                    term_degrees[j] = 0
                } else {
                    term_degrees[j] += 1
                    break
                }
            }
        }

        res += polynomial_value
    }

    volume : f32 = 1.0
    for i in 0..<number_of_variables {
        volume *= region[2*i + 1] - region[2*i]
    }

    res = volume * res / f32(n)
    return res
}

// https://github.com/odin-lang/examples/blob/master/thread/basics/thread_basics.odin
Thread_Data :: struct {
    number_of_variables: i32,
    max_degrees: [^]i32,
    coefficients: [^]f32,
    region: [^]f32,
    n: i32,
    res: f32,
}

// TODO:
// Check thread safety of rand
// Will cause issue if n < 8 due to division by zero - fix this
@export
polynomial_monte_carlo :: proc "c" (
    number_of_variables: i32,
    max_degrees: [^]i32,
    coefficients: [^]f32,
    region: [^]f32,
    n: i32,
) -> f32 {
    context = runtime.default_context()

    threads_data : [7]Thread_Data
    for i in 0..<7 {
        threads_data[i] = Thread_Data{
            number_of_variables = number_of_variables,
            max_degrees = max_degrees,
            coefficients = coefficients,
            region = region,
            n = n / 8,
            res = 0,
        }
    }

    // If we join a thread before starting it may not run: https://github.com/odin-lang/Odin/issues/3622
    // Solution from laytan from Odin Discord: Use a wait group
    started : sync.Wait_Group
    sync.wait_group_add(&started, 7)
    threads : [7]^thread.Thread
    for i in 0..<7 {
        threads[i] = thread.create_and_start_with_poly_data2(
            &started,
            &threads_data[i],
            proc(started: ^sync.Wait_Group, data: ^Thread_Data) {
                sync.wait_group_done(started)
                data.res = polynomial_monte_carlo_single_thread(data.number_of_variables, data.max_degrees, data.coefficients, data.region, data.n)
            }
        )
    }
    sync.wait_group_wait(&started)

    res := polynomial_monte_carlo_single_thread(number_of_variables, max_degrees, coefficients, region, n - 7 * (n / 8)) * f32(n - 7 * (n / 8))

    for i in 0..<7 {
        thread.join(threads[i])
        thread.destroy(threads[i])
    }

    for i in 0..<7 {
        res += threads_data[i].res * f32(threads_data[i].n)
    }
    return res / f32(n)
}