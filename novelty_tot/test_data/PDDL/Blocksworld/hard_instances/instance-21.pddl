

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a c)
(on b f)
(on c b)
(ontable d)
(ontable e)
(on f e)
(clear a)
(clear d)
)
(:goal
(and
(on a e)
(on d c)
(on e f))
)
)


