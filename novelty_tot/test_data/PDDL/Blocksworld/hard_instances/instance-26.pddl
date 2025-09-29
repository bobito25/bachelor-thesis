

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a c)
(ontable b)
(on c d)
(on d e)
(on e f)
(ontable f)
(clear a)
(clear b)
)
(:goal
(and
(on c f)
(on e c)
(on f a))
)
)


